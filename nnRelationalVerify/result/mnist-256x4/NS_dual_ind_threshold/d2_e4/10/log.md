## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 23.708826041400002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808)
1: (-18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632)
2: (-22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983)
3: (-25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631)
4: (-23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213)
5: (-18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036)
6: (-19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349)
7: (-23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605)
8: (-27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121)
9: (-16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.02 + 12.57 = 14.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -23.7325583, upper bound: 23.7325576

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7312597, upper bound: 23.7314442
time: 8.37 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7318928, upper bound: 23.7318922
time: 27.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 36.24 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 36.24
Output dim: 1, lower bound: -23.7312597, upper bound: 23.7314442
NS_A2, status: Status.UNKNOWN, split count: 1, time: 36.24
Output dim: 1, lower bound: -23.7318928, upper bound: 23.7318922

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -17.3089657, 15.6176853, -18.1656303, 16.3576012, -33.6665649, 33.7833176
1: -16.8956890, 10.6321707, -17.7043343, 11.2288599, -28.1245499, 28.3365021
2: -20.3298264, 13.2015057, -21.3376408, 13.8329468, -34.1627731, 34.5391464
3: -24.0127907, 11.5651112, -25.1294250, 12.1181622, -36.1309509, 36.6945343
4: -22.0331421, 14.0476408, -23.0579491, 14.7386856, -36.7718277, 37.1055908
5: -16.8948975, 15.0914574, -17.7348137, 15.7933502, -32.6882439, 32.8262634
6: -17.8104057, 16.2330074, -18.6470718, 17.0265923, -34.8369980, 34.8800812
7: -21.5968590, 15.5667610, -22.5833607, 16.3253822, -37.9222412, 38.1501236
8: -25.7235413, 12.9762993, -26.8873901, 13.6666155, -39.3901520, 39.8636856
9: -15.6413260, 17.6513081, -16.4399548, 18.4842339, -34.1255569, 34.0912552

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7301682, upper bound: 23.7305279
time: 6.22 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7300536, upper bound: 23.7301858
time: 5.90 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -17.9763165, 16.1946182, -18.4789600, 16.6207371, -34.5970535, 34.6735764
1: -17.5284500, 11.0939922, -17.9937820, 11.4514093, -28.9798584, 29.0877743
2: -21.1167107, 13.6932383, -21.7028847, 14.0653181, -35.1820297, 35.3961182
3: -24.8852310, 11.9967546, -25.5235558, 12.3169203, -37.2021484, 37.5203056
4: -22.8374977, 14.5846176, -23.4217834, 14.9921703, -37.8296661, 38.0064011
5: -17.5454063, 15.6410904, -18.0413303, 16.0436020, -33.5890083, 33.6824188
6: -18.4643784, 16.8513908, -18.9463921, 17.3141193, -35.7784958, 35.7977829
7: -22.3681087, 16.1552162, -22.9342976, 16.6016350, -38.9697418, 39.0895157
8: -26.6359844, 13.5126390, -27.2989349, 13.9225731, -40.5585556, 40.8115730
9: -16.2623425, 18.3026600, -16.7342205, 18.7794361, -35.0417786, 35.0368805

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7306320, upper bound: 23.7308898
time: 5.11 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7305071, upper bound: 23.7305071
time: 6.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 13.16 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 13.16
Output dim: 1, lower bound: -23.7301682, upper bound: 23.7305279
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 13.16
Output dim: 1, lower bound: -23.7300536, upper bound: 23.7301858
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 13.16
Output dim: 1, lower bound: -23.7306320, upper bound: 23.7308898
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 13.16
Output dim: 1, lower bound: -23.7305071, upper bound: 23.7305071

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -17.1545753, 15.4796104, -17.5679741, 15.8243885, -32.9789619, 33.0475807
1: -16.7493820, 10.5393410, -17.1409225, 10.8691349, -27.6185169, 27.6802597
2: -20.1477795, 13.0864735, -20.6345215, 13.3892727, -33.5370483, 33.7209930
3: -23.7991199, 11.4647284, -24.3037090, 11.7313623, -35.5304794, 35.7684364
4: -21.8396416, 13.9236383, -22.3137512, 14.2563705, -36.0960083, 36.2373886
5: -16.7440567, 14.9586849, -17.1481018, 15.2834663, -32.0275230, 32.1067772
6: -17.6527176, 16.0901375, -18.0387306, 16.4748077, -34.1275177, 34.1288605
7: -21.4068546, 15.4303837, -21.8511162, 15.7960682, -37.2029228, 37.2815018
8: -25.4963512, 12.8595161, -26.0098190, 13.2157173, -38.7120667, 38.8693275
9: -15.5017757, 17.4968796, -15.8995447, 17.8891792, -33.3909531, 33.3964233

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7297372, upper bound: 23.7300915
time: 8.98 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7297255, upper bound: 23.7300961
time: 11.07 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -16.9441986, 15.2915535, -18.7247753, 16.8477268, -33.7919235, 34.0163269
1: -16.5510540, 10.4136915, -18.2156868, 11.5599651, -28.1110172, 28.6293736
2: -19.8995190, 12.9292965, -21.9920979, 14.2410784, -34.1405907, 34.9213943
3: -23.5083923, 11.3281250, -25.9339447, 12.4719563, -35.9803467, 37.2620659
4: -21.5764656, 13.7546520, -23.7909336, 15.1922646, -36.7687225, 37.5455818
5: -16.5386658, 14.7778730, -18.2806511, 16.2778091, -32.8164711, 33.0585175
6: -17.4378471, 15.8956680, -19.2293701, 17.5479488, -34.9857903, 35.1250381
7: -21.1475830, 15.2444916, -23.2806244, 16.8343658, -37.9819489, 38.5251122
8: -25.1865921, 12.7004948, -27.6991196, 14.0583363, -39.2449265, 40.3996124
9: -15.3119259, 17.2866192, -16.9573536, 19.0493736, -34.3612900, 34.2439651

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 196

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7296162, upper bound: 23.7297375
time: 5.40 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7296073, upper bound: 23.7297384
time: 9.46 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -17.8190842, 16.0542698, -17.8775024, 16.0854454, -33.9045296, 33.9317589
1: -17.3801918, 10.9993124, -17.4281178, 11.0889730, -28.4691582, 28.4274292
2: -20.9316978, 13.5765543, -20.9960251, 13.6184816, -34.5501709, 34.5725746
3: -24.6679745, 11.8949232, -24.6947975, 11.9282703, -36.5962448, 36.5897217
4: -22.6416740, 14.4577503, -22.6735687, 14.5073442, -37.1490173, 37.1313171
5: -17.3909607, 15.5069637, -17.4520035, 15.5308428, -32.9218025, 32.9589691
6: -18.3043251, 16.7061977, -18.3353386, 16.7599487, -35.0642700, 35.0415344
7: -22.1754189, 16.0159645, -22.1991291, 16.0692825, -38.2446976, 38.2150841
8: -26.4051075, 13.3939819, -26.4186153, 13.4689589, -39.8740654, 39.8125916
9: -16.1201496, 18.1460609, -16.1899281, 18.1822319, -34.3023796, 34.3359909

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 107

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7301976, upper bound: 23.7304573
time: 8.86 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7301920, upper bound: 23.7304605
time: 7.70 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -17.6045113, 15.8621464, -19.0577126, 17.1267967, -34.7313080, 34.9198608
1: -17.1783791, 10.8700409, -18.5187798, 11.7985916, -28.9769630, 29.3888206
2: -20.6779423, 13.4161968, -22.3838615, 14.4938803, -35.1718216, 35.8000526
3: -24.3718605, 11.7556953, -26.3535023, 12.6826115, -37.0544662, 38.1091995
4: -22.3744316, 14.2844038, -24.1796398, 15.4592400, -37.8336716, 38.4640427
5: -17.1801338, 15.3236628, -18.6067448, 16.5447197, -33.7248535, 33.9304085
6: -18.0852909, 16.5079460, -19.5488110, 17.8546524, -35.9399376, 36.0567551
7: -21.9117317, 15.8255730, -23.6559544, 17.1274719, -39.0392036, 39.4815216
8: -26.0893917, 13.2313423, -28.1349144, 14.3314991, -40.4208908, 41.3662567
9: -15.9256716, 17.9319592, -17.2723427, 19.3599224, -35.2855911, 35.2042923

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7300694, upper bound: 23.7300665
time: 5.06 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7300670, upper bound: 23.7300668
time: 6.28 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 13.30 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.30
Output dim: 1, lower bound: -23.7297372, upper bound: 23.7300915
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.30
Output dim: 1, lower bound: -23.7297255, upper bound: 23.7300961
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.30
Output dim: 1, lower bound: -23.7296162, upper bound: 23.7297375
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.30
Output dim: 1, lower bound: -23.7296073, upper bound: 23.7297384
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.30
Output dim: 1, lower bound: -23.7301976, upper bound: 23.7304573
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.30
Output dim: 1, lower bound: -23.7301920, upper bound: 23.7304605
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.30
Output dim: 1, lower bound: -23.7300694, upper bound: 23.7300665
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.30
Output dim: 1, lower bound: -23.7300670, upper bound: 23.7300668

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -15.6758175, 14.1854877, -17.0316391, 15.3543749, -31.0301933, 31.2171268
1: -15.3361893, 9.5596218, -16.6300926, 10.5041494, -25.8403358, 26.1897125
2: -18.3675594, 11.9870405, -19.9871712, 12.9883003, -31.3558598, 31.9742126
3: -21.8137245, 10.5056858, -23.5914574, 11.3841581, -33.1978798, 34.0971451
4: -20.0140820, 12.7246399, -21.6566372, 13.8165274, -33.8306046, 34.3812752
5: -15.2930126, 13.7180634, -16.6141510, 14.8377056, -30.1307182, 30.3322144
6: -16.1667137, 14.7059402, -17.5012627, 15.9736319, -32.1403465, 32.2071991
7: -19.6078415, 14.1140194, -21.2039833, 15.3137407, -34.9215736, 35.3180008
8: -23.4069214, 11.7017002, -25.2595139, 12.7904005, -36.1973228, 36.9612122
9: -14.1433926, 16.0149364, -15.3998747, 17.3555412, -31.4989338, 31.4148102

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 136

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7267731, upper bound: 23.7276766
time: 10.48 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7266154, upper bound: 23.7272598
time: 6.31 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -16.3841629, 14.8085489, -17.3166618, 15.6054878, -31.9896507, 32.1252060
1: -16.0155277, 10.0182962, -16.9034824, 10.6978550, -26.7133827, 26.9217796
2: -19.2269802, 12.5156288, -20.3347168, 13.2024298, -32.4294090, 32.8503456
3: -22.7843914, 10.9665213, -23.9748344, 11.5699444, -34.3543358, 34.9413528
4: -20.9003849, 13.2997675, -22.0101242, 14.0504522, -34.9508362, 35.3098907
5: -15.9854393, 14.3167458, -16.8981819, 15.0764523, -31.0618877, 31.2149258
6: -16.8889179, 15.3731613, -17.7901993, 16.2413521, -33.1302719, 33.1633606
7: -20.4860573, 14.7490864, -21.5529785, 15.5716858, -36.0577393, 36.3020630
8: -24.4263039, 12.2474318, -25.6619568, 13.0160046, -37.4423027, 37.9093857
9: -14.7947960, 16.7325020, -15.6669884, 17.6420250, -32.4368172, 32.3994904

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 112

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7268972, upper bound: 23.7277494
time: 4.85 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7267313, upper bound: 23.7273490
time: 6.38 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -15.5208025, 14.0481777, -18.2230701, 16.4142437, -31.9350414, 32.2712440
1: -15.1898384, 9.4653578, -17.7406425, 11.2144184, -26.4042568, 27.2060013
2: -18.1846390, 11.8720694, -21.3905220, 13.8676758, -32.0523109, 33.2625923
3: -21.5992641, 10.4060583, -25.2785816, 12.1500845, -33.7493439, 35.6846352
4: -19.8210468, 12.6020641, -23.1832886, 14.7785053, -34.5995483, 35.7853470
5: -15.1429825, 13.5863199, -17.7834358, 15.8643513, -31.0073338, 31.3697548
6: -16.0097885, 14.5629549, -18.7319565, 17.0807667, -33.0905533, 33.2949104
7: -19.4177666, 13.9785385, -22.6836529, 16.3842125, -35.8019791, 36.6621895
8: -23.1820068, 11.5859709, -27.0092659, 13.6572132, -36.8392181, 38.5952377
9: -14.0048695, 15.8605747, -16.4879532, 18.5550518, -32.5599213, 32.3485260

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 112

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7266543, upper bound: 23.7273923
time: 88.71 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7264855, upper bound: 23.7268756
time: 5.02 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -16.2114983, 14.6553469, -18.4921265, 16.6491165, -32.8606148, 33.1474724
1: -15.8530540, 9.9134970, -17.9976025, 11.3994970, -27.2525501, 27.9110985
2: -19.0238533, 12.3870316, -21.7173347, 14.0688906, -33.0927429, 34.1043625
3: -22.5483932, 10.8544827, -25.6366692, 12.3245335, -34.8729210, 36.4911499
4: -20.6863937, 13.1612720, -23.5138817, 15.0009050, -35.6872940, 36.6751480
5: -15.8176842, 14.1694756, -18.0520153, 16.0877495, -31.9054298, 32.2214890
6: -16.7138615, 15.2138176, -19.0025406, 17.3333397, -34.0472031, 34.2163582
7: -20.2753868, 14.5975533, -23.0100060, 16.6279869, -36.9033699, 37.6075592
8: -24.1746120, 12.1152344, -27.3846779, 13.8714752, -38.0460854, 39.4999123
9: -14.6392269, 16.5609837, -16.7410240, 18.8234482, -33.4626732, 33.3020096

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 112

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7295969, upper bound: 23.7297295
time: 4.98 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7296073, upper bound: 23.7297384
time: 57.70 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -16.3248310, 14.7406473, -17.3140450, 15.5943823, -31.9192123, 32.0546913
1: -15.9542398, 10.0007105, -16.8961315, 10.7056494, -26.6598892, 26.8968430
2: -19.1339264, 12.4623718, -20.3189316, 13.1947403, -32.3286667, 32.7813034
3: -22.6641903, 10.9263668, -23.9501419, 11.5665617, -34.2307510, 34.8765068
4: -20.7962303, 13.2424335, -21.9876022, 14.0444345, -34.8406601, 35.2300339
5: -15.9117165, 14.2530661, -16.8912220, 15.0648022, -30.9765186, 31.1442871
6: -16.8004055, 15.3117256, -17.7733421, 16.2347145, -33.0351181, 33.0850677
7: -20.3595428, 14.6801567, -21.5225773, 15.5629988, -35.9225388, 36.2027321
8: -24.2990112, 12.2218819, -25.6344509, 13.0226078, -37.3216171, 37.8563309
9: -14.7389812, 16.6509151, -15.6643810, 17.6264191, -32.3653984, 32.3152924

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 112

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7281269, upper bound: 23.7287816
time: 5.28 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7279610, upper bound: 23.7283396
time: 4.80 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -17.0317402, 15.3645096, -17.6103134, 15.8542690, -32.8860054, 32.9748230
1: -16.6300297, 10.4636335, -17.1778393, 10.9068012, -27.5368252, 27.6414719
2: -19.9872665, 12.9902639, -20.6784744, 13.4176798, -33.4049416, 33.6687317
3: -23.6302147, 11.3852921, -24.3473167, 11.7582235, -35.3884392, 35.7326088
4: -21.6796608, 13.8173180, -22.3517208, 14.2886868, -35.9683456, 36.1690369
5: -16.6112556, 14.8507900, -17.1875000, 15.3108826, -31.9221382, 32.0382919
6: -17.5210285, 15.9731331, -18.0724182, 16.5120850, -34.0331116, 34.0455475
7: -21.2342606, 15.3150396, -21.8834057, 15.8311815, -37.0654411, 37.1984406
8: -25.3093834, 12.7672310, -26.0514660, 13.2566166, -38.5659981, 38.8186951
9: -15.3923016, 17.3643188, -15.9416342, 17.9219418, -33.3142357, 33.3059540

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7281544, upper bound: 23.7288070
time: 9.24 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7279944, upper bound: 23.7283543
time: 9.29 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -16.1637497, 14.5979242, -18.5282059, 16.6703682, -32.8341179, 33.1261292
1: -15.8023129, 9.9019251, -18.0219288, 11.4331388, -27.2354507, 27.9238510
2: -18.9445305, 12.3422985, -21.7502022, 14.0955753, -33.0401001, 34.0924988
3: -22.4450474, 10.8217697, -25.6650047, 12.3449163, -34.7899628, 36.4867668
4: -20.5970631, 13.1130295, -23.5395813, 15.0237093, -35.6207733, 36.6526108
5: -15.7552624, 14.1159010, -18.0823479, 16.1093636, -31.8646259, 32.1982498
6: -16.6374855, 15.1630707, -19.0262642, 17.3614769, -33.9989548, 34.1893349
7: -20.1637478, 14.5389099, -23.0287189, 16.6521568, -36.8159027, 37.5676270
8: -24.0651894, 12.0976381, -27.4102707, 13.9082680, -37.9734573, 39.5079041
9: -14.5936546, 16.4911346, -16.7746124, 18.8432426, -33.4368935, 33.2657471

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 112

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7300679, upper bound: 23.7300644
time: 12.09 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7300694, upper bound: 23.7300668
time: 8.19 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -16.8556900, 15.2079372, -18.8137531, 16.9180374, -33.7737274, 34.0216904
1: -16.4640007, 10.3565292, -18.2917080, 11.6295881, -28.0935898, 28.6482372
2: -19.7801399, 12.8587933, -22.0952301, 14.3111067, -34.0912399, 34.9540253
3: -23.3894482, 11.2710648, -26.0419331, 12.5283918, -35.9178314, 37.3129959
4: -21.4611206, 13.6759100, -23.8886147, 15.2594948, -36.7206154, 37.5645218
5: -16.4397488, 14.7002954, -18.3663254, 16.3456841, -32.7854309, 33.0666199
6: -17.3424149, 15.8107471, -19.3116627, 17.6289444, -34.9713593, 35.1224060
7: -21.0191383, 15.1601610, -23.3724995, 16.9106312, -37.9297714, 38.5326576
8: -25.0528679, 12.6324492, -27.8056717, 14.1357613, -39.1886292, 40.4381218
9: -15.2333832, 17.1893082, -17.0445709, 19.1248741, -34.3582573, 34.2338791

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7300637, upper bound: 23.7300643
time: 5.28 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7300671, upper bound: 23.7300668
time: 5.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 12.67 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.67
Output dim: 1, lower bound: -23.7267731, upper bound: 23.7276766
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.67
Output dim: 1, lower bound: -23.7266154, upper bound: 23.7272598
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.67
Output dim: 1, lower bound: -23.7268972, upper bound: 23.7277494
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.67
Output dim: 1, lower bound: -23.7267313, upper bound: 23.7273490
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.67
Output dim: 1, lower bound: -23.7266543, upper bound: 23.7273923
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.67
Output dim: 1, lower bound: -23.7264855, upper bound: 23.7268756
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.67
Output dim: 1, lower bound: -23.7295969, upper bound: 23.7297295
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.67
Output dim: 1, lower bound: -23.7296073, upper bound: 23.7297384
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.67
Output dim: 1, lower bound: -23.7281269, upper bound: 23.7287816
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.67
Output dim: 1, lower bound: -23.7279610, upper bound: 23.7283396
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.67
Output dim: 1, lower bound: -23.7281544, upper bound: 23.7288070
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.67
Output dim: 1, lower bound: -23.7279944, upper bound: 23.7283543
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.67
Output dim: 1, lower bound: -23.7300679, upper bound: 23.7300644
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.67
Output dim: 1, lower bound: -23.7300694, upper bound: 23.7300668
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.67
Output dim: 1, lower bound: -23.7300637, upper bound: 23.7300643
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.67
Output dim: 1, lower bound: -23.7300671, upper bound: 23.7300668

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -15.5810165, 14.1032343, -16.5646725, 14.9437943, -30.5248108, 30.6679039
1: -15.2477045, 9.5008745, -16.1940231, 10.2108507, -25.4585552, 25.6948967
2: -18.2570763, 11.9181614, -19.4448643, 12.6454048, -30.9024811, 31.3630257
3: -21.6845913, 10.4445457, -22.9592094, 11.0818510, -32.7664413, 33.4037552
4: -19.8969822, 12.6501369, -21.0788784, 13.4458828, -33.3428574, 33.7290154
5: -15.2021332, 13.6385136, -16.1568871, 14.4409580, -29.6430893, 29.7954006
6: -16.0721951, 14.6182594, -17.0346375, 15.5440998, -31.6162949, 31.6528969
7: -19.4945297, 14.0313644, -20.6465149, 14.9024773, -34.3970070, 34.6778755
8: -23.2724361, 11.6298513, -24.6000843, 12.4317913, -35.7042274, 36.2299309
9: -14.0572414, 15.9211979, -14.9709740, 16.8935204, -30.9507618, 30.8921719

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7257922, upper bound: 23.7263439
time: 5.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7253186, upper bound: 23.7261837
time: 5.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -15.4158554, 13.9625340, -17.9546642, 16.1981354, -31.6139889, 31.9171982
1: -15.0945530, 9.3945599, -17.4997540, 10.9896002, -26.0841522, 26.8943138
2: -18.0655479, 11.7997189, -21.0980244, 13.6778183, -31.7433662, 32.8977432
3: -21.4634438, 10.3383999, -24.9688549, 11.9809628, -33.4444046, 35.3072548
4: -19.6958427, 12.5224295, -22.8714294, 14.5587626, -34.2546043, 35.3938484
5: -15.0450382, 13.5025930, -17.5132103, 15.6532974, -30.6983356, 31.0158043
6: -15.9104490, 14.4660473, -18.4862518, 16.8325520, -32.7429962, 32.9523010
7: -19.3023186, 13.8881741, -22.4220734, 16.1364441, -35.4387627, 36.3102455
8: -23.0449848, 11.5032873, -26.7162819, 13.4222488, -36.4672318, 38.2195702
9: -13.9066267, 15.7592621, -16.2091370, 18.3039169, -32.2105370, 31.9683914

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7255656, upper bound: 23.7257774
time: 6.35 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7251305, upper bound: 23.7256766
time: 5.37 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -16.2873325, 14.7245483, -16.8439693, 15.1898060, -31.4771347, 31.5685177
1: -15.9252796, 9.9580307, -16.4625473, 10.4010544, -26.3263340, 26.4205761
2: -19.1145973, 12.4452839, -19.7854271, 12.8554916, -31.9700890, 32.2307129
3: -22.6536446, 10.9039097, -23.3349876, 11.2640409, -33.9176865, 34.2388992
4: -20.7815704, 13.2234001, -21.4257469, 13.6750832, -34.4566536, 34.6491432
5: -15.8919172, 14.2354069, -16.4351521, 14.6754093, -30.5673199, 30.6705589
6: -16.7925377, 15.2839556, -17.3177605, 15.8066254, -32.5991592, 32.6017151
7: -20.3708801, 14.6647129, -20.9891281, 15.1551552, -35.5260277, 35.6538391
8: -24.2895679, 12.1734142, -24.9941120, 12.6530380, -36.9426041, 37.1675262
9: -14.7067738, 16.6370239, -15.2324886, 17.1748428, -31.8816166, 31.8695068

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7243614, upper bound: 23.7245620
time: 5.89 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7232679, upper bound: 23.7241940
time: 5.69 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -16.1165867, 14.5787621, -18.2799397, 16.4865494, -32.6031342, 32.8586998
1: -15.7673416, 9.8481770, -17.8117714, 11.2050133, -26.9723530, 27.6599445
2: -18.9172859, 12.3220215, -21.4941368, 13.9224968, -32.8397827, 33.8161583
3: -22.4271851, 10.7935009, -25.4115181, 12.1934443, -34.6206284, 36.2050133
4: -20.5747528, 13.0895138, -23.2781315, 14.8249950, -35.3997498, 36.3676453
5: -15.7286205, 14.0945625, -17.8366947, 15.9283895, -31.6570072, 31.9312515
6: -16.6252098, 15.1264648, -18.8182526, 17.1380234, -33.7632332, 33.9447174
7: -20.1724682, 14.5164833, -22.8244572, 16.4303684, -36.6028366, 37.3409348
8: -24.0543442, 12.0401726, -27.1822834, 13.6759806, -37.7303238, 39.2224503
9: -14.5501404, 16.4700851, -16.5117359, 18.6328220, -33.1829605, 32.9818115

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7241075, upper bound: 23.7239304
time: 6.41 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7230499, upper bound: 23.7236228
time: 6.12 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -15.4260635, 13.9660873, -17.7336521, 15.9856701, -31.4117336, 31.6997395
1: -15.1014137, 9.4066906, -17.2852135, 10.9081221, -26.0095367, 26.6919041
2: -18.0742321, 11.8034725, -20.8244686, 13.5104971, -31.5847263, 32.6279411
3: -21.4700718, 10.3450727, -24.6177158, 11.8348560, -33.3049240, 34.9627876
4: -19.7039738, 12.5280037, -22.5830269, 14.3882093, -34.0921822, 35.1110229
5: -15.0522308, 13.5068378, -17.3027725, 15.4521065, -30.5043335, 30.8096104
6: -15.9153557, 14.4755611, -18.2452908, 16.6319141, -32.5472679, 32.7208481
7: -19.3046799, 13.8959560, -22.1024895, 15.9526892, -35.2573700, 35.9984436
8: -23.0476494, 11.5145798, -26.3183327, 13.2834167, -36.3310623, 37.8329124
9: -13.9190197, 15.7668819, -16.0398083, 18.0747757, -31.9937954, 31.8066902

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7256849, upper bound: 23.7261004
time: 7.91 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7251997, upper bound: 23.7259345
time: 9.43 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -15.2603807, 13.8249445, -19.0954151, 17.2146587, -32.4750404, 32.9203568
1: -14.9477177, 9.3000422, -18.5640793, 11.6729946, -26.6207123, 27.8641205
2: -17.8821716, 11.6846447, -22.4458427, 14.5232172, -32.4053879, 34.1304855
3: -21.2483292, 10.2386208, -26.5854073, 12.7162533, -33.9645844, 36.8240280
4: -19.5022774, 12.3999033, -24.3389874, 15.4790344, -34.9813118, 36.7388878
5: -14.8946152, 13.3704729, -18.6320114, 16.6393623, -31.5339756, 32.0024834
6: -15.7531052, 14.3229055, -19.6680222, 17.8951130, -33.6482162, 33.9909286
7: -19.1117344, 13.7523127, -23.8428364, 17.1631966, -36.2749252, 37.5951500
8: -22.8194466, 11.3877029, -28.3920746, 14.2553644, -37.0748062, 39.7797775
9: -13.7680454, 15.6044674, -17.2545090, 19.4571648, -33.2252121, 32.8589745

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254550, upper bound: 23.7254875
time: 5.31 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7249946, upper bound: 23.7253466
time: 7.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -15.8534698, 14.3567276, -17.3914890, 15.7207355, -31.5742054, 31.7482166
1: -15.5110731, 9.6554108, -16.9566574, 10.6273165, -26.1383896, 26.6120682
2: -18.6075745, 12.1265726, -20.4419556, 13.2667217, -31.8742962, 32.5685272
3: -22.0952454, 10.6271973, -24.2308674, 11.6245213, -33.7197647, 34.8580589
4: -20.2636795, 12.8751001, -22.2104225, 14.1112432, -34.3749237, 35.0855217
5: -15.4820194, 13.8792353, -17.0013466, 15.1920223, -30.6740417, 30.8805809
6: -16.3719711, 14.8835926, -17.9385147, 16.3237438, -32.6957169, 32.8221054
7: -19.8715477, 14.2888031, -21.7585793, 15.6700783, -35.5416260, 36.0473824
8: -23.7023525, 11.8260231, -25.9091949, 12.9896679, -36.6920204, 37.7352142
9: -14.3155451, 16.2139225, -15.7383289, 17.7621346, -32.0776787, 31.9522514

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7275839, upper bound: 23.7273435
time: 11.01 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7266013, upper bound: 23.7269489
time: 7.32 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -16.0191326, 14.4944172, -17.8721218, 16.1291008, -32.1482315, 32.3665352
1: -15.6699476, 9.7761021, -17.4134102, 10.9598475, -26.6297951, 27.1895084
2: -18.8009930, 12.2473288, -21.0027504, 13.6182909, -32.4192848, 33.2500763
3: -22.3043556, 10.7318850, -24.8539715, 11.9301872, -34.2345390, 35.5858574
4: -20.4587593, 13.0076599, -22.7851639, 14.4990654, -34.9578209, 35.7928162
5: -15.6366005, 14.0129013, -17.4595566, 15.5860825, -31.2226830, 31.4724541
6: -16.5299091, 15.0369568, -18.4081173, 16.7661476, -33.2960587, 33.4450760
7: -20.0582390, 14.4319868, -22.3129139, 16.0897465, -36.1479874, 36.7448997
8: -23.9202995, 11.9599075, -26.5625610, 13.3693399, -37.2896347, 38.5224686
9: -14.4653301, 16.3744755, -16.1749649, 18.2290611, -32.6943893, 32.5494385

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7275844, upper bound: 23.7273435
time: 6.50 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7266013, upper bound: 23.7269486
time: 11.16 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -16.2269974, 14.6558456, -16.8309822, 15.1714697, -31.3984680, 31.4868279
1: -15.8629646, 9.9395943, -16.4475002, 10.4018097, -26.2647743, 26.3870945
2: -19.0204048, 12.3912916, -19.7590847, 12.8420219, -31.8624268, 32.1503716
3: -22.5322514, 10.8630800, -23.2984962, 11.2549505, -33.7872009, 34.1615753
4: -20.6763039, 13.1652756, -21.3949318, 13.6592369, -34.3355331, 34.5602074
5: -15.8171768, 14.1709805, -16.4159622, 14.6581173, -30.4752941, 30.5869427
6: -16.7030773, 15.2215824, -17.2923412, 15.7916393, -32.4947166, 32.5139236
7: -20.2433777, 14.5949059, -20.9489632, 15.1365662, -35.3799438, 35.5438690
8: -24.1610889, 12.1469679, -24.9546089, 12.6522064, -36.8132935, 37.1015778
9: -14.6500301, 16.5544853, -15.2205858, 17.1521339, -31.8021641, 31.7750664

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7269493, upper bound: 23.7273067
time: 6.92 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7264322, upper bound: 23.7271349
time: 5.89 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16.0652008, 14.5176096, -18.2229652, 16.4279041, -32.4931030, 32.7405739
1: -15.7128506, 9.8349857, -17.7544861, 11.1828146, -26.8956623, 27.5894718
2: -18.8333664, 12.2743368, -21.4157066, 13.8766689, -32.7100372, 33.6900444
3: -22.3179131, 10.7584181, -25.3111038, 12.1556129, -34.4735222, 36.0695229
4: -20.4804039, 13.0382442, -23.1909370, 14.7739220, -35.2543144, 36.2291794
5: -15.6621428, 14.0375776, -17.7745514, 15.8724422, -31.5345840, 31.8121262
6: -16.5444736, 15.0722389, -18.7466240, 17.0817261, -33.6261940, 33.8188629
7: -20.0552921, 14.4543486, -22.7274590, 16.3724937, -36.4277878, 37.1818047
8: -23.9382133, 12.0204353, -27.0737686, 13.6447239, -37.5829391, 39.0942039
9: -14.5016060, 16.3962479, -16.4616070, 18.5647030, -33.0663071, 32.8578568

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7255656, upper bound: 23.7267155
time: 7.60 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7262461, upper bound: 23.7265893
time: 5.67 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -16.9319248, 15.2769108, -17.1117611, 15.4186993, -32.3506241, 32.3886719
1: -16.5368881, 10.4007778, -16.7165794, 10.5935555, -27.1304436, 27.1173573
2: -19.8714981, 12.9170904, -20.1021271, 13.0537825, -32.9252815, 33.0192184
3: -23.4952908, 11.3206873, -23.6758881, 11.4380522, -34.9333420, 34.9965668
4: -21.5563259, 13.7381668, -21.7434082, 13.8895378, -35.4458618, 35.4815750
5: -16.5135422, 14.7660894, -16.6958046, 14.8934402, -31.4069786, 31.4618950
6: -17.4214439, 15.8813782, -17.5775185, 16.0553284, -33.4767723, 33.4588966
7: -21.1153450, 15.2271862, -21.2931919, 15.3904982, -36.5058403, 36.5203781
8: -25.1688042, 12.6905003, -25.3504467, 12.8750610, -38.0438652, 38.0409393
9: -15.3006306, 17.2656231, -15.4840651, 17.4344101, -32.7350388, 32.7496796

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7255398, upper bound: 23.7255734
time: 7.09 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7245047, upper bound: 23.7252447
time: 8.71 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -16.7601547, 15.1265965, -18.5498581, 16.7175102, -33.4776649, 33.6764526
1: -16.3769665, 10.2878618, -18.0672913, 11.3999071, -27.7768688, 28.3551521
2: -19.6731873, 12.7909012, -21.8146095, 14.1228943, -33.7960815, 34.6055031
3: -23.2670555, 11.2092018, -25.7562809, 12.3690920, -35.6361465, 36.9654846
4: -21.3458729, 13.6018810, -23.5991573, 15.0415726, -36.3874435, 37.2010345
5: -16.3440571, 14.6211004, -18.1001625, 16.1484985, -32.4925537, 32.7212639
6: -17.2521191, 15.7236958, -19.0809498, 17.3885708, -34.6406898, 34.8046455
7: -20.9142990, 15.0753880, -23.1319351, 16.6680069, -37.5823059, 38.2073212
8: -24.9328918, 12.5553455, -27.5418358, 13.9002800, -38.8331718, 40.0971832
9: -15.1406593, 17.0964260, -16.7663364, 18.8951626, -34.0358200, 33.8627625

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7252615, upper bound: 23.7248205
time: 6.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7242634, upper bound: 23.7245294
time: 5.98 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 14.59 + 588.64 = 603.22 seconds
