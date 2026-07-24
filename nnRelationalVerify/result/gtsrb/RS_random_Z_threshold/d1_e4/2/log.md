## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 1800 seconds
Split limit: 100
Threshold: 12.5603180091


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9000626, 13.9000664)
1: (-3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5087318, 8.5087318)
2: (-0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4614334, 13.4614372)
3: (-1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0319748, 12.0319729)
4: (-11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6887894, 14.6887856)
5: (1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796)
6: (-39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2155037, 15.2154999)
7: (-3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6391602, 13.6391640)
8: (-6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1128159, 12.1128178)
9: (-4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0374336, 13.0374336)
10: (1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9535446, 20.9535446)
11: (-11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476)
12: (-11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0364494, 15.0364494)
13: (-18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6293907, 16.6293945)
14: (4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7749405, 26.7749405)
15: (-8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198)
16: (-16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8331146, 14.8331184)
17: (6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2550125, 17.2550125)
18: (-14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4317722, 14.4317741)
19: (-20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5550842, 14.5550919)
20: (-2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6365433, 12.6365433)
21: (-11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549)
22: (-3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9717941, 14.9717941)
23: (-14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3402557, 14.3402557)
24: (-19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2769775, 9.2769775)
25: (-5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8234940, 13.8234940)
26: (-21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3724327, 19.3724365)
27: (-16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2311211, 13.2311211)
28: (-12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141)
29: (-5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9777222, 14.9777222)
30: (-10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5620842, 13.5620842)
31: (-10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6624298, 14.6624260)
32: (-24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3205376, 13.3205414)
33: (-69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6724014, 16.6724014)
34: (-53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1694450, 14.1694450)
35: (-47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0219498, 13.0219536)
36: (-42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1241531, 15.1241570)
37: (-86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9252739, 18.9252701)
38: (-52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3915100, 18.3915100)
39: (-76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0968819, 16.0968819)
40: (-67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3520813, 14.3520813)
41: (-55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7036438, 16.7036476)
42: (-29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2716637, 17.2716675)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.77 + 20.69 = 23.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 14, lower bound: -12.5728910, upper bound: 12.5728909

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 938

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 997

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5727244, upper bound: 12.5623201
time: 12.18 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5623201, upper bound: 12.5727243
time: 43.43 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 55.63 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 55.63
Output dim: 14, lower bound: -12.5727244, upper bound: 12.5623201
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 55.63
Output dim: 14, lower bound: -12.5623201, upper bound: 12.5727243

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8999367, 13.8998718
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5086670, 8.5086193
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4612427, 13.4616585
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0315475, 12.0312176
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6878662, 14.6883125
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2151680, 15.2152939
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6390686, 13.6390953
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1110840, 12.1118088
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0369377, 13.0364609
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9527130, 20.9520798
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0360718, 15.0351143
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6270332, 16.6253395
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7736359, 26.7729111
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8331070, 14.8330917
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2542229, 17.2528381
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4313030, 14.4322224
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5539246, 14.5544395
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6355247, 12.6360359
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9717102, 14.9717369
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3393402, 14.3398743
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2754745, 9.2759132
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8234024, 13.8233719
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3693886, 19.3704681
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2251816, 13.2266121
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9776840, 14.9777069
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5623550, 13.5619850
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6616974, 14.6622543
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3202744, 13.3202934
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6717224, 16.6719093
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1690636, 14.1693039
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0218468, 13.0219040
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1236877, 15.1238441
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9246635, 18.9249535
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3905640, 18.3910217
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0966110, 16.0967255
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3519096, 14.3522377
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7014465, 16.7021484
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2714272, 17.2714386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 770

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 979

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5724747, upper bound: 12.5506470
time: 11.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5610672, upper bound: 12.5620695
time: 24.47 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8998718, 13.8999367
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5086212, 8.5086651
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4616547, 13.4612427
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0312195, 12.0315495
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6883087, 14.6878624
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2152901, 15.2151680
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6390991, 13.6390648
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1118088, 12.1110840
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0364609, 13.0369377
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9520798, 20.9527130
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0351143, 15.0360718
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6253395, 16.6270370
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7729034, 26.7736359
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8330917, 14.8331032
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2528343, 17.2542229
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4322224, 14.4313030
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5544434, 14.5539169
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6360397, 12.6355247
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9717331, 14.9717102
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3398743, 14.3393440
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2759132, 9.2754707
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8233719, 13.8234024
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3704643, 19.3693924
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2266121, 13.2251816
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9777069, 14.9776878
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5619850, 13.5623550
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6622543, 14.6616974
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3202896, 13.3202744
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6719055, 16.6717262
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1693077, 14.1690636
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0219078, 13.0218468
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1238403, 15.1236877
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9249535, 18.9246674
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3910217, 18.3905640
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0967255, 16.0966110
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3522377, 14.3519096
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7021484, 16.7014465
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2714348, 17.2714233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1728

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 976

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5622758, upper bound: 12.5680659
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5576472, upper bound: 12.5726804
time: 6.99 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.01 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.01
Output dim: 14, lower bound: -12.5724747, upper bound: 12.5506470
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.01
Output dim: 14, lower bound: -12.5610672, upper bound: 12.5620695
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.01
Output dim: 14, lower bound: -12.5622758, upper bound: 12.5680659
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.01
Output dim: 14, lower bound: -12.5576472, upper bound: 12.5726804

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8979797, 13.8974838
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5065231, 8.5060577
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4606094, 13.4609337
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0277290, 12.0269470
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6888962, 14.6895370
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2149429, 15.2150688
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6358414, 13.6354332
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1074104, 12.1076622
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0346107, 13.0338478
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9488831, 20.9479599
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0361214, 15.0351639
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6291313, 16.6276627
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7699432, 26.7688141
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8278313, 14.8267441
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2469406, 17.2446594
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4283619, 14.4294930
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5513840, 14.5521927
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6331825, 12.6340141
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9675064, 14.9681854
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3388901, 14.3393974
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2722359, 9.2729988
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8234596, 13.8235054
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3627625, 19.3648682
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2209396, 13.2227135
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9748230, 14.9752159
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5625954, 13.5623245
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6617050, 14.6622620
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3184280, 13.3184280
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6666412, 16.6671028
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1648560, 14.1655579
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0148392, 13.0156746
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1102943, 15.1119156
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9135208, 18.9144135
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3762512, 18.3783188
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0885925, 16.0896339
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3522758, 14.3524609
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6958084, 16.6966515
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2736702, 17.2732391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1290

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1594

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5714929, upper bound: 12.5467687
time: 11.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5686197, upper bound: 12.5496597
time: 6.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8975487, 13.8979111
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5061035, 8.5064774
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4605179, 13.4610291
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0272789, 12.0273933
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6890945, 14.6893501
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2149429, 15.2150650
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6354065, 13.6358757
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1069374, 12.1081390
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0343285, 13.0341339
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9486008, 20.9482498
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0361176, 15.0351639
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6293602, 16.6274300
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7695312, 26.7692261
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8267479, 14.8278236
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2460480, 17.2455559
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4285755, 14.4292812
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5516739, 14.5518990
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6334991, 12.6336937
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9681549, 14.9675331
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3388748, 14.3394241
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2725601, 9.2726746
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8235359, 13.8234253
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3638000, 19.3638382
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2212868, 13.2223701
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9752045, 14.9748383
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5626907, 13.5622292
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6617050, 14.6622620
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3184090, 13.3184471
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6669159, 16.6668205
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1653137, 14.1650963
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0156174, 13.0148964
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1117592, 15.1104469
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9141312, 18.9138069
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3778610, 18.3767090
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0895157, 16.0887108
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3521309, 14.3526039
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6959534, 16.6965103
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2732277, 17.2736816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1574

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 947

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5608580, upper bound: 12.5618569
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5608550, upper bound: 12.5618595
time: 7.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8997307, 13.8997803
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5091743, 8.5092812
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4614258, 13.4608612
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0304337, 12.0306129
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6882019, 14.6881866
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2140274, 15.2139511
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6389847, 13.6389389
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1089363, 12.1078663
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0373268, 13.0377808
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9504852, 20.9505386
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0332489, 15.0344200
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6168022, 16.6193466
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7719269, 26.7724991
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8327408, 14.8327103
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2500572, 17.2510719
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4325218, 14.4316177
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5542259, 14.5538292
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6360130, 12.6355362
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9718056, 14.9719810
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3389740, 14.3380737
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2754211, 9.2750359
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8236771, 13.8237572
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3707275, 19.3696442
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2259636, 13.2245445
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9766922, 14.9765511
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5613098, 13.5615921
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6623230, 14.6619186
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3178520, 13.3181343
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6621971, 16.6632385
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1654129, 14.1656685
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0130005, 13.0138702
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1153870, 15.1162262
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9202805, 18.9206200
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3801880, 18.3810120
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0817413, 16.0833817
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3519363, 14.3515930
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7005463, 16.7000618
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2730255, 17.2727509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 952

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 988

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5620783, upper bound: 12.5679873
time: 8.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5621974, upper bound: 12.5678692
time: 28.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8997116, 13.8997955
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5092354, 8.5092201
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4612808, 13.4610100
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0302811, 12.0307617
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6886292, 14.6877594
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2140732, 15.2139015
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6389694, 13.6389503
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1085930, 12.1082134
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0373039, 13.0378036
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9499054, 20.9511261
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0334625, 15.0342064
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6176567, 16.6184921
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7717743, 26.7726593
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8327026, 14.8327484
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2496834, 17.2514420
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4325371, 14.4316025
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5543480, 14.5537071
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6360474, 12.6354980
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9720039, 14.9717789
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3386078, 14.3384476
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2754745, 9.2749825
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8237228, 13.8237076
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3707199, 19.3696594
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2259750, 13.2245369
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9765701, 14.9766693
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5612221, 13.5616760
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6624756, 14.6617661
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3181534, 13.3178329
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6634254, 16.6620140
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1659088, 14.1651726
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0139313, 13.0129433
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1163788, 15.1152382
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9209061, 18.9199867
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3814697, 18.3797226
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0835037, 16.0816269
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3519211, 14.3516083
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7007599, 16.6998558
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2727585, 17.2730141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 675

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 933

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5480490, upper bound: 12.5721583
time: 20.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5571185, upper bound: 12.5631153
time: 10.47 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 32.93 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 32.93
Output dim: 14, lower bound: -12.5714929, upper bound: 12.5467687
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 32.93
Output dim: 14, lower bound: -12.5686197, upper bound: 12.5496597
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 32.93
Output dim: 14, lower bound: -12.5608580, upper bound: 12.5618569
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 32.93
Output dim: 14, lower bound: -12.5608550, upper bound: 12.5618595
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 32.93
Output dim: 14, lower bound: -12.5620783, upper bound: 12.5679873
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 32.93
Output dim: 14, lower bound: -12.5621974, upper bound: 12.5678692
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 32.93
Output dim: 14, lower bound: -12.5480490, upper bound: 12.5721583
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 32.93
Output dim: 14, lower bound: -12.5571185, upper bound: 12.5631153

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8982048, 13.8970871
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5065308, 8.5059929
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4615822, 13.4608154
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0269661, 12.0274448
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6889191, 14.6894951
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2127609, 15.2161484
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6358109, 13.6353760
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1091614, 12.1072388
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0346069, 13.0333748
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9501877, 20.9476852
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0346565, 15.0357132
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6268425, 16.6293106
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7698669, 26.7683487
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8289413, 14.8255348
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2471237, 17.2438393
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4283295, 14.4287395
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5515709, 14.5516891
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6327286, 12.6341896
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9674683, 14.9690208
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3394318, 14.3385429
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2727356, 9.2727737
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8234444, 13.8234901
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3633728, 19.3641968
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2207489, 13.2223740
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9745636, 14.9749451
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5627556, 13.5622787
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6616936, 14.6622467
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3163223, 13.3196106
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6642685, 16.6680603
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1629372, 14.1667023
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0129852, 13.0172615
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1083336, 15.1148758
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9133072, 18.9144287
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3713226, 18.3811417
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0850258, 16.0903969
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3524895, 14.3524246
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6949844, 16.6966553
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2732353, 17.2729874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 690

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1294

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5710329, upper bound: 12.5467257
time: 11.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5714504, upper bound: 12.5463064
time: 6.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8975868, 13.8974838
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5064621, 8.5060577
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4604988, 13.4609337
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0277290, 12.0261860
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6888580, 14.6895370
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2149429, 15.2128868
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6358414, 13.6354027
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1069946, 12.1076622
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0341339, 13.0338478
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9486084, 20.9479599
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0361214, 15.0336990
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6291313, 16.6253815
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7694702, 26.7688141
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8266296, 14.8267441
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2461166, 17.2446594
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4276085, 14.4294930
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5508766, 14.5521927
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6331825, 12.6335564
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9675064, 14.9681473
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3380356, 14.3393974
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2720108, 9.2729988
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8234596, 13.8234901
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3620834, 19.3648682
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2205963, 13.2227135
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9745483, 14.9752159
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5625496, 13.5623245
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6617050, 14.6622543
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3184280, 13.3163223
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6666412, 16.6647339
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1648560, 14.1636353
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0148392, 13.0138168
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1102943, 15.1099548
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9135208, 18.9141998
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3762512, 18.3733978
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0885925, 16.0860672
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3522377, 14.3524609
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6958084, 16.6958237
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2736702, 17.2728043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1283

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5490242, upper bound: 12.5493515
time: 14.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5683108, upper bound: 12.5300806
time: 41.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8976479, 13.8977356
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5058517, 8.5062943
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4601746, 13.4606247
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0259438, 12.0270424
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6881409, 14.6877785
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2141342, 15.2144394
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6343765, 13.6348419
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1062965, 12.1082230
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0339470, 13.0336342
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9459152, 20.9468842
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0356979, 15.0346107
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6270332, 16.6235962
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7687378, 26.7692184
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8264732, 14.8275337
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2460022, 17.2455139
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4282513, 14.4292393
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5503540, 14.5506973
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6332817, 12.6337090
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9676437, 14.9670639
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3383255, 14.3391342
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2690239, 9.2696533
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8229637, 13.8231430
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3633423, 19.3634491
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2207146, 13.2220764
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9745178, 14.9743690
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5613365, 13.5611801
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6616402, 14.6622391
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3184052, 13.3183784
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6654587, 16.6637878
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1643257, 14.1640739
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0155640, 13.0146980
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1119843, 15.1095467
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9120483, 18.9097061
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3779793, 18.3764572
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0854187, 16.0820007
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3484688, 14.3471947
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6960030, 16.6956940
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2722740, 17.2723694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 980

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5607303, upper bound: 12.5563662
time: 8.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5578037, upper bound: 12.5618362
time: 5.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8973732, 13.8980103
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5059204, 8.5062256
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4601135, 13.4606895
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0269279, 12.0260544
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6875153, 14.6884041
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2143173, 15.2142487
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6343765, 13.6348495
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1070290, 12.1074982
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0338326, 13.0337524
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9472351, 20.9455643
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0355644, 15.0347404
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6255226, 16.6251068
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7695160, 26.7684326
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8264656, 14.8275414
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2460022, 17.2455063
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4285336, 14.4289589
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5504684, 14.5505829
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6335144, 12.6334763
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9676895, 14.9670219
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3385849, 14.3388748
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2695389, 9.2691383
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8232536, 13.8228493
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3634186, 19.3633728
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2209930, 13.2217979
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9747314, 14.9741631
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5616455, 13.5608749
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6616859, 14.6621933
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3183403, 13.3184433
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6638870, 16.6653595
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1642952, 14.1641006
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0154190, 13.0148468
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1108551, 15.1106758
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9100266, 18.9117279
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3776131, 18.3768234
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0828094, 16.0846100
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3467216, 14.3489418
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6951332, 16.6965599
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2719154, 17.2727318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 952

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 899

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5532455, upper bound: 12.5617724
time: 11.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5607680, upper bound: 12.5542504
time: 9.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9057922, 13.9050293
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5090981, 8.5088387
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4617386, 13.4612694
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0312119, 12.0314331
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6884232, 14.6884308
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2081909, 15.2072906
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6385117, 13.6384468
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1150665, 12.1149902
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0453873, 13.0443077
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9500961, 20.9501419
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0282211, 15.0277596
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6096115, 16.6090927
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7734146, 26.7734680
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8261871, 14.8249855
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2461319, 17.2466431
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4290085, 14.4286194
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5527191, 14.5525284
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6367416, 12.6368294
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9662933, 14.9668999
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3323975, 14.3323517
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2701530, 9.2705879
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8176498, 13.8183632
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3707581, 19.3701477
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2147026, 13.2150574
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9731941, 14.9734459
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5613098, 13.5615959
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6599312, 14.6606293
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3162003, 13.3162842
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6644974, 16.6647797
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1618690, 14.1621284
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0157013, 13.0163765
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1153641, 15.1161728
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9245567, 18.9239731
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3801880, 18.3810272
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0705719, 16.0711517
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3487549, 14.3481255
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7000580, 16.6994286
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2733078, 17.2730827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 956

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 945

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5620397, upper bound: 12.5516940
time: 9.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5457738, upper bound: 12.5679488
time: 11.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9049797, 13.9058418
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5087318, 8.5092049
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4618378, 13.4611778
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0312500, 12.0313950
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6884537, 14.6884003
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2073669, 15.2081184
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6384888, 13.6384773
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1160583, 12.1139946
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0438538, 13.0458412
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9500885, 20.9501495
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0265884, 15.0293922
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6065445, 16.6121635
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7729111, 26.7739792
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8250122, 14.8261566
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2456207, 17.2471466
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4295235, 14.4281044
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5529327, 14.5523186
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6373062, 12.6362648
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9667206, 14.9664726
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3332520, 14.3314934
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2709770, 9.2697678
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8182755, 13.8177299
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3712387, 19.3696747
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2164764, 13.2132835
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9735832, 14.9730568
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5613136, 13.5615959
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6610374, 14.6595268
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3159981, 13.3164825
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6637344, 16.6655426
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1618767, 14.1621208
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0155029, 13.0165672
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1153412, 15.1161995
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9236259, 18.9249001
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3802032, 18.3810120
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0695114, 16.0722122
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3484688, 14.3484116
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6999207, 16.6995621
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2733612, 17.2730255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 977

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1288

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5614909, upper bound: 12.5677456
time: 13.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5620740, upper bound: 12.5671618
time: 9.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8972511, 13.8971977
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5025368, 8.5030193
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4585800, 13.4582329
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0263824, 12.0272007
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6885986, 14.6877213
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2117615, 15.2118683
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6320877, 13.6324768
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1080704, 12.1084938
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0421486, 13.0421371
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9493713, 20.9506302
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0253105, 15.0259857
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6298981, 16.6286201
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7715836, 26.7724838
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8218155, 14.8225365
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2517738, 17.2539024
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4334450, 14.4325714
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5543137, 14.5535927
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6353874, 12.6349335
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9659576, 14.9651299
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3369217, 14.3368378
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2780685, 9.2775688
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8258629, 13.8263245
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3661575, 19.3646088
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2224884, 13.2209892
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9749527, 14.9750977
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5606308, 13.5616684
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6607361, 14.6602631
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3198586, 13.3192520
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6470566, 16.6437035
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1634521, 14.1624069
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0079765, 13.0056190
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1158752, 15.1127968
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9055252, 18.9026413
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3752594, 18.3725891
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0619926, 16.0575867
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3450279, 14.3437080
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6980057, 16.6964378
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2733002, 17.2735939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1292

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5427990, upper bound: 12.5720496
time: 8.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5479393, upper bound: 12.5669350
time: 6.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8971138, 13.8973351
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5030365, 8.5025196
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4585037, 13.4583130
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0267181, 12.0268688
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6885986, 14.6877251
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2120438, 15.2115860
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6324921, 13.6320686
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1088715, 12.1076927
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0416336, 13.0426483
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9494095, 20.9505920
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0252419, 15.0260544
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6277771, 16.6307373
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7715988, 26.7724609
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8224869, 14.8218727
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2521477, 17.2535324
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4335060, 14.4325104
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5542374, 14.5536652
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6354828, 12.6348419
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9653625, 14.9657326
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3369904, 14.3367691
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2780609, 9.2775764
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8263359, 13.8258400
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3656693, 19.3651047
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2224274, 13.2210503
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9749985, 14.9750557
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5612144, 13.5610847
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6609726, 14.6600227
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3195724, 13.3195381
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6451035, 16.6456413
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1631393, 14.1627197
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0066032, 13.0069923
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1139374, 15.1147308
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9035568, 18.9046097
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3743286, 18.3735199
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0594597, 16.0601196
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3440208, 14.3447151
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6973419, 16.6971016
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2733383, 17.2735519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 774

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 838

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5467650, upper bound: 12.5605202
time: 33.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5545234, upper bound: 12.5527615
time: 6.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 42.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.50
Output dim: 14, lower bound: -12.5710329, upper bound: 12.5467257
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.50
Output dim: 14, lower bound: -12.5714504, upper bound: 12.5463064
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 42.50
Output dim: 14, lower bound: -12.5490242, upper bound: 12.5493515
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.50
Output dim: 14, lower bound: -12.5683108, upper bound: 12.5300806
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.50
Output dim: 14, lower bound: -12.5607303, upper bound: 12.5563662
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.50
Output dim: 14, lower bound: -12.5578037, upper bound: 12.5618362
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.50
Output dim: 14, lower bound: -12.5532455, upper bound: 12.5617724
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.50
Output dim: 14, lower bound: -12.5607680, upper bound: 12.5542504
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.50
Output dim: 14, lower bound: -12.5620397, upper bound: 12.5516940
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.50
Output dim: 14, lower bound: -12.5457738, upper bound: 12.5679488
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.50
Output dim: 14, lower bound: -12.5614909, upper bound: 12.5677456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.50
Output dim: 14, lower bound: -12.5620740, upper bound: 12.5671618
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.50
Output dim: 14, lower bound: -12.5427990, upper bound: 12.5720496
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.50
Output dim: 14, lower bound: -12.5479393, upper bound: 12.5669350
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.50
Output dim: 14, lower bound: -12.5467650, upper bound: 12.5605202
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 42.50
Output dim: 14, lower bound: -12.5545234, upper bound: 12.5527615

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8983192, 13.8971672
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5067062, 8.5061398
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4618225, 13.4610519
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0270004, 12.0274773
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6889877, 14.6895561
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2125931, 15.2159767
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6362762, 13.6358070
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1092873, 12.1073914
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0341339, 13.0329056
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9492798, 20.9468307
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0344620, 15.0355301
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6267052, 16.6291580
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7698822, 26.7683563
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8288879, 14.8254852
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2471275, 17.2438698
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4278088, 14.4281616
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5513458, 14.5514221
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6324692, 12.6339302
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9677238, 14.9693184
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3386574, 14.3377228
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2722969, 9.2723083
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8234482, 13.8235016
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3633308, 19.3641510
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2207947, 13.2224083
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9747620, 14.9751816
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5624580, 13.5620193
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6611404, 14.6616936
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3164291, 13.3196983
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6651154, 16.6688232
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1636391, 14.1674385
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0136566, 13.0178680
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1083565, 15.1148987
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9137840, 18.9148636
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3703690, 18.3801041
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0851860, 16.0905342
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3517189, 14.3515892
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6950989, 16.6967506
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2733650, 17.2730980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1287

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 977

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5705065, upper bound: 12.5346581
time: 14.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5589937, upper bound: 12.5461946
time: 6.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8982849, 13.8972015
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5066795, 8.5061665
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4618225, 13.4610519
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0270004, 12.0274773
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6889801, 14.6895638
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2125854, 15.2159843
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6362457, 13.6358414
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1093178, 12.1073685
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0341377, 13.0329018
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9493408, 20.9467697
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0344772, 15.0355186
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6266975, 16.6291733
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7698822, 26.7683563
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8288879, 14.8254776
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2471504, 17.2438469
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4277515, 14.4282207
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5513077, 14.5514641
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6324654, 12.6339340
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9677620, 14.9692764
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3386116, 14.3377686
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2722702, 9.2723312
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8234482, 13.8235016
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3633308, 19.3641510
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2207832, 13.2224197
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9748001, 14.9751434
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5624962, 13.5619850
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6611404, 14.6616974
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3164101, 13.3197174
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6650314, 16.6689072
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1636696, 14.1674042
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0135880, 13.0179367
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1083565, 15.1148987
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9137383, 18.9149094
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3702927, 18.3801804
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0851631, 16.0905533
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3516579, 14.3516502
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6950836, 16.6967659
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2733421, 17.2731171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 704

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 899

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5638419, upper bound: 12.5462196
time: 10.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5713633, upper bound: 12.5386963
time: 6.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8953133, 13.8952179
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4968796, 8.4951134
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4525452, 13.4521027
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0233574, 12.0219460
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6824722, 14.6824417
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2048492, 15.2036171
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6204224, 13.6175537
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0998383, 12.0998383
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0326996, 13.0318756
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9629517, 20.9641953
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0291176, 15.0273476
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6363258, 16.6326141
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7586594, 26.7566681
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8261948, 14.8257637
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2419624, 17.2393684
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4267273, 14.4290924
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5485191, 14.5499268
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6465225, 12.6454659
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9658737, 14.9663239
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3271446, 14.3295975
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2566605, 9.2584686
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8237686, 13.8237762
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3363304, 19.3413925
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2312698, 13.2325325
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9638290, 14.9648819
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5545998, 13.5546646
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6746445, 14.6731453
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3236961, 13.3218117
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6669083, 16.6681709
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1193886, 14.1230240
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9833794, 12.9858780
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0894814, 15.0915070
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8497047, 18.8573265
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3557167, 18.3554535
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.1072426, 16.1074562
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3250427, 14.3285713
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6644058, 16.6679993
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2641220, 17.2640419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 773

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 768

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5683035, upper bound: 12.5295168
time: 35.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5677531, upper bound: 12.5300732
time: 9.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8971481, 13.8963737
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5053940, 8.5052986
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4580688, 13.4582748
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0251541, 12.0262222
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6868744, 14.6865463
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2147675, 15.2150841
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6298828, 13.6297150
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1043625, 12.1061249
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0337448, 13.0330315
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9473648, 20.9482880
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0355568, 15.0344658
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6250191, 16.6214600
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7699890, 26.7704239
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8218002, 14.8218536
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2463341, 17.2457504
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4273987, 14.4284744
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5500641, 14.5504417
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6326714, 12.6334343
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9618835, 14.9621315
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3382263, 14.3390388
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2674637, 9.2683449
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8207397, 13.8212738
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3572540, 19.3581543
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2203751, 13.2217636
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9672623, 14.9680214
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5586319, 13.5588951
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6615295, 14.6621437
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3196144, 13.3192825
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6661911, 16.6642761
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1637993, 14.1635971
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0157928, 13.0149422
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1110725, 15.1087456
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9134293, 18.9109077
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3782425, 18.3767853
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0817566, 16.0778847
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3416939, 14.3395920
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6989975, 16.6985817
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2751884, 17.2749176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1574

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 952

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5240298, upper bound: 12.5549030
time: 11.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5592698, upper bound: 12.5196576
time: 8.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8962860, 13.8972359
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5048561, 8.5058365
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4578323, 13.4585190
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0251236, 12.0262604
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6869125, 14.6865234
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2147751, 15.2150650
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6292496, 13.6303520
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1042023, 12.1062851
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0333443, 13.0334320
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9473419, 20.9483337
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0355492, 15.0344734
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6248970, 16.6215820
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7699432, 26.7704620
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8207855, 14.8228645
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2462349, 17.2458420
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4274864, 14.4283848
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5501099, 14.5504074
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6330070, 12.6330986
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9627151, 14.9613075
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3382263, 14.3390388
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2677116, 9.2680969
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8210907, 13.8209190
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3580399, 19.3573608
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2204018, 13.2217331
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9681778, 14.9671059
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5590591, 13.5584755
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6615448, 14.6621246
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3193092, 13.3195877
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6659470, 16.6646729
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1638298, 14.1635551
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0158081, 13.0149307
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1111794, 15.1086311
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9132538, 18.9111481
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3782959, 18.3767242
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0812988, 16.0783424
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3408699, 14.3404217
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6988907, 16.6987534
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2748222, 17.2752838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 838

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5474523, upper bound: 12.5592459
time: 10.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5552098, upper bound: 12.5514883
time: 8.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8910294, 13.8925323
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4987373, 8.4999084
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4574738, 13.4583778
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0253372, 12.0246449
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6861801, 14.6869888
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2076416, 15.2065659
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6310425, 13.6319809
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1048813, 12.1059341
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0296173, 13.0300751
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9431839, 20.9420013
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0323372, 15.0313454
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6293526, 16.6288414
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7632446, 26.7628403
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8163414, 14.8191223
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2479095, 17.2477837
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4228382, 14.4237823
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5535202, 14.5540543
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6301613, 12.6296196
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9640732, 14.9624367
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3418808, 14.3427467
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2705650, 9.2700920
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8249779, 13.8248329
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3626099, 19.3624649
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2193489, 13.2196655
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9710197, 14.9698029
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5608559, 13.5600319
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6647415, 14.6655922
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3090668, 13.3078308
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6533241, 16.6534653
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1546783, 14.1530914
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0114365, 13.0103149
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1018524, 15.1000900
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9048882, 18.9058571
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3685760, 18.3667068
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0751190, 16.0757408
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3396568, 14.3408184
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6898308, 16.6905289
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2676697, 17.2679977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1339

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1594

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5522620, upper bound: 12.5578992
time: 8.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5493730, upper bound: 12.5607873
time: 6.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8918953, 13.8916702
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4996033, 8.4990463
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4578018, 13.4580498
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0255127, 12.0244656
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6861038, 14.6870689
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2066345, 15.2075691
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6315079, 13.6315155
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1054611, 12.1053543
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0301552, 13.0295372
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9436722, 20.9415131
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0321693, 15.0315170
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6292610, 16.6289406
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7639313, 26.7621689
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8180428, 14.8174248
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2482834, 17.2474098
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4233570, 14.4232635
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5539322, 14.5536308
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6296577, 12.6301231
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9631042, 14.9634094
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3424454, 14.3421783
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2704926, 9.2701607
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8252373, 13.8245659
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3625031, 19.3625641
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2188606, 13.2201538
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9703636, 14.9704514
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5608025, 13.5600853
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6650848, 14.6652489
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3077278, 13.3091736
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6519966, 16.6547966
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1532822, 14.1544876
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0108871, 13.0108604
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1002731, 15.1016693
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9041557, 18.9065933
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3674927, 18.3677826
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0739365, 16.0769196
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3386002, 14.3418751
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6890984, 16.6912537
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2671738, 17.2684898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 856

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5537261, upper bound: 12.5541736
time: 8.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5606899, upper bound: 12.5472066
time: 6.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9082870, 13.9073677
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5080490, 8.5084057
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4613228, 13.4607964
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0357094, 12.0368690
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6824188, 14.6832771
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1957664, 15.1964302
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6381149, 13.6388741
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1161499, 12.1160221
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0448341, 13.0436859
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9531250, 20.9527130
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0231552, 15.0233269
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6108551, 16.6103210
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7639160, 26.7625885
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8283920, 14.8272362
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2378807, 17.2371979
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4306374, 14.4304237
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5527840, 14.5526085
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6366119, 12.6367302
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9641075, 14.9647217
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3277130, 14.3269386
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2782059, 9.2772865
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8125191, 13.8116150
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3670502, 19.3641739
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2132072, 13.2136574
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9640617, 14.9629364
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5587959, 13.5580559
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6594391, 14.6605721
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3038254, 13.3063354
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6413231, 16.6444893
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1407471, 14.1438103
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0012779, 13.0044060
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1154175, 15.1166763
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9233818, 18.9228668
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3656693, 18.3683319
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0552292, 16.0576782
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3429375, 14.3424225
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6921539, 16.6931648
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2726135, 17.2722626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 840

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5196197, upper bound: 12.5515112
time: 8.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5618570, upper bound: 12.5092702
time: 16.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9081306, 13.9075241
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5086670, 8.5077877
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4612694, 13.4608498
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0366478, 12.0359268
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6832733, 14.6824303
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1973381, 15.1948662
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6389465, 13.6380463
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1160965, 12.1160717
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0447617, 13.0437584
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9526672, 20.9531631
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0237923, 15.0226898
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6108398, 16.6103363
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7625427, 26.7639542
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8284302, 14.8271904
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2366829, 17.2383919
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4308128, 14.4302483
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5527992, 14.5525970
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6366425, 12.6367035
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9641228, 14.9647064
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3269882, 14.3276634
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2768517, 9.2786407
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8109016, 13.8132286
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3647842, 19.3664398
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2133026, 13.2135620
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9626884, 14.9643097
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5577736, 13.5590820
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6598740, 14.6601372
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3062515, 13.3039093
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6442070, 16.6415939
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1435471, 14.1410065
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0037270, 13.0019569
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1158676, 15.1162262
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9234505, 18.9228020
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3674927, 18.3665161
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0570984, 16.0558052
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3430519, 14.3423080
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6937943, 16.6915207
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2724915, 17.2723923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 704

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1003

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5456985, upper bound: 12.5673615
time: 13.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5451661, upper bound: 12.5678684
time: 7.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9046822, 13.9055023
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5083466, 8.5088787
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4618988, 13.4612389
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0315094, 12.0315571
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6869431, 14.6870995
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2053146, 15.2058411
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6385956, 13.6385803
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1163483, 12.1141853
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0439072, 13.0458527
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9500504, 20.9500732
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0259018, 15.0285988
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6055603, 16.6113319
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7725525, 26.7736588
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8251572, 14.8262749
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2455978, 17.2471237
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4289494, 14.4276352
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5528259, 14.5524216
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6372375, 12.6361961
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9669075, 14.9667473
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3332901, 14.3315353
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2707024, 9.2697296
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8186378, 13.8182564
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3715439, 19.3700714
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2164726, 13.2132797
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9736938, 14.9731979
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5612488, 13.5615959
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6606865, 14.6593018
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3145103, 13.3147430
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6637497, 16.6655464
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1615181, 14.1617393
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0158157, 13.0169945
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1149902, 15.1158066
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9236183, 18.9248886
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3804703, 18.3812256
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0690308, 16.0717583
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3465576, 14.3463097
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6985474, 16.6980705
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2700424, 17.2694168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1712

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5190742, upper bound: 12.5675630
time: 11.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5613083, upper bound: 12.5253284
time: 29.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9046402, 13.9055443
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5084000, 8.5088177
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4618912, 13.4612503
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0314102, 12.0316525
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6871490, 14.6868973
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2050858, 15.2060623
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6385956, 13.6385841
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1162567, 12.1142769
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0438652, 13.0458946
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9500122, 20.9501114
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0257950, 15.0287056
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6057129, 16.6111832
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7725830, 26.7736359
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8251419, 14.8262939
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2455978, 17.2471199
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4290562, 14.4275322
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5530319, 14.5522118
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6372375, 12.6361961
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9669991, 14.9666443
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3332977, 14.3315277
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2709389, 9.2694931
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8188057, 13.8180809
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3716354, 19.3699799
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2164726, 13.2132797
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9737244, 14.9731598
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5613136, 13.5615311
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6608086, 14.6591759
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3142586, 13.3149948
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6637497, 16.6655502
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1614952, 14.1617661
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0159302, 13.0168800
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1149445, 15.1158524
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9236183, 18.9248886
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3804169, 18.3812866
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0690536, 16.0717316
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3463669, 14.3465042
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6984253, 16.6981926
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2697525, 17.2697067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 972

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 950

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5430454, upper bound: 12.5612089
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5561293, upper bound: 12.5481235
time: 8.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9010391, 13.9005165
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5022087, 8.5026855
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4587479, 13.4583817
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0229874, 12.0241547
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6900787, 14.6890488
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9141541, 15.9159126
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2123871, 15.2124748
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6299896, 13.6306114
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1076355, 12.1080799
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0494080, 13.0485916
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9462433, 20.9471359
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0202179, 15.0205307
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6323700, 16.6309090
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7677231, 26.7691193
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8245697, 14.8241844
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2503242, 17.2526207
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4332695, 14.4326077
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5544853, 14.5537758
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6383705, 12.6388092
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9625549, 14.9620705
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3378754, 14.3377762
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2844696, 9.2842216
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8316422, 13.8333168
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3674622, 19.3659439
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2134666, 13.2125816
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9711075, 14.9716797
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5628433, 13.5642242
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6621017, 14.6625214
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3248940, 13.3238106
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6279144, 16.6221428
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1684532, 14.1660652
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0081749, 13.0047684
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1165466, 15.1134148
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8792648, 18.8733177
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3751793, 18.3730774
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0496368, 16.0434380
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3247871, 14.3206100
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6953659, 16.6920700
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2697220, 17.2695618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5224022, upper bound: 12.5718640
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5426123, upper bound: 12.5516785
time: 17.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9005699, 13.9009857
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5022049, 8.5026894
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4587326, 13.4584007
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0233383, 12.0238075
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6899261, 14.6892014
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9149246, 15.9151344
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2123566, 15.2125015
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6302261, 13.6303749
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1076584, 12.1080570
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0486031, 13.0493965
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9458771, 20.9475021
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0198555, 15.0208931
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6321869, 16.6310997
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7682266, 26.7686234
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8234711, 14.8252831
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2504921, 17.2524529
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4334793, 14.4323978
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5544853, 14.5537682
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6392632, 12.6379204
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9628983, 14.9617310
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3378677, 14.3377876
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2847214, 9.2839699
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8328476, 13.8321114
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3674927, 19.3659134
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2140808, 13.2119675
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9715347, 14.9712524
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5631866, 13.5638809
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6629944, 14.6616325
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3244209, 13.3242836
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6254883, 16.6245689
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1671104, 14.1674080
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0071297, 13.0058136
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1165009, 15.1134605
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8761978, 18.8763771
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3757515, 18.3725052
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0478516, 16.0452271
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3219261, 14.3234692
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6936417, 16.6937981
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2692642, 17.2700195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1291

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 779

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5478769, upper bound: 12.5643783
time: 21.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5453826, upper bound: 12.5668726
time: 13.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8970833, 13.8973160
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5029678, 8.5032845
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4582748, 13.4585075
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0268631, 12.0267506
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6882172, 14.6880608
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2126541, 15.2106590
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6323700, 13.6326866
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1086121, 12.1084557
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0419960, 13.0421371
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9495621, 20.9503937
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0259247, 15.0247879
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6298447, 16.6305351
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7714767, 26.7724075
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8224258, 14.8224945
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2518578, 17.2534485
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4329033, 14.4331799
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5536880, 14.5541458
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6353683, 12.6349678
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9654922, 14.9657249
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3366852, 14.3370247
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2770691, 9.2782784
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8254585, 13.8262672
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3656311, 19.3651047
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2223206, 13.2220879
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9749985, 14.9750557
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5610619, 13.5616150
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6599960, 14.6607437
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3204422, 13.3181610
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6469841, 16.6451988
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1641388, 14.1617966
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0077820, 13.0068855
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1150703, 15.1138840
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9042816, 18.9031792
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3748856, 18.3727493
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0608139, 16.0596504
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3445358, 14.3436108
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6979942, 16.6960716
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2739105, 17.2726707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 980

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5467440, upper bound: 12.5550407
time: 12.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5412878, upper bound: 12.5604991
time: 55.93 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 70.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5705065, upper bound: 12.5346581
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5589937, upper bound: 12.5461946
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5638419, upper bound: 12.5462196
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5713633, upper bound: 12.5386963
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5683035, upper bound: 12.5295168
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5677531, upper bound: 12.5300732
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5240298, upper bound: 12.5549030
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5592698, upper bound: 12.5196576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5474523, upper bound: 12.5592459
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5552098, upper bound: 12.5514883
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5522620, upper bound: 12.5578992
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5493730, upper bound: 12.5607873
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5537261, upper bound: 12.5541736
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5606899, upper bound: 12.5472066
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5196197, upper bound: 12.5515112
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5618570, upper bound: 12.5092702
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5456985, upper bound: 12.5673615
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5451661, upper bound: 12.5678684
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5190742, upper bound: 12.5675630
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5613083, upper bound: 12.5253284
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5430454, upper bound: 12.5612089
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5561293, upper bound: 12.5481235
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5224022, upper bound: 12.5718640
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5426123, upper bound: 12.5516785
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5478769, upper bound: 12.5643783
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5453826, upper bound: 12.5668726
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5467440, upper bound: 12.5550407
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 70.51
Output dim: 14, lower bound: -12.5412878, upper bound: 12.5604991

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8956757, 13.8942032
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5063248, 8.5057316
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4598618, 13.4587784
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0233765, 12.0233803
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6914749, 14.6928749
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2068214, 15.2106247
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6343765, 13.6336212
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1030197, 12.1004066
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0344009, 13.0331421
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9396286, 20.9361038
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0306282, 15.0322266
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6216774, 16.6259079
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7648621, 26.7625046
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8233452, 14.8194046
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2360802, 17.2312660
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4278660, 14.4282227
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5484161, 14.5487633
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6296806, 12.6314468
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9628983, 14.9652596
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3405685, 14.3389015
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2711563, 9.2712479
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8236618, 13.8237190
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3605843, 19.3615875
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2207642, 13.2225952
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9747009, 14.9751472
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5623055, 13.5614128
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6598511, 14.6605568
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3073692, 13.3117561
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6416550, 16.6483002
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1519775, 14.1572495
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9930344, 12.9998131
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0849991, 15.0944214
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8969955, 18.9002075
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3428497, 18.3555069
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0522270, 16.0616493
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3506355, 14.3505325
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6864204, 16.6891708
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2756653, 17.2749634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 780

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5704547, upper bound: 12.5329799
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5688353, upper bound: 12.5346068
time: 8.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8919411, 13.8917236
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4994965, 8.4998493
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4591637, 13.4587250
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0254097, 12.0260601
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6876450, 14.6881523
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2059021, 15.2083015
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6329041, 13.6329613
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1071701, 12.1058044
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0299301, 13.0292282
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9453201, 20.9432373
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0312462, 15.0321198
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6305122, 16.6328888
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7635956, 26.7627487
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8187790, 14.8170624
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2490578, 17.2461281
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4220600, 14.4230480
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5543518, 14.5549393
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6291046, 12.6300659
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9641533, 14.9646912
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3419189, 14.3416443
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2732964, 9.2732849
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8251801, 13.8254929
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3624878, 19.3632126
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2191467, 13.2202988
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9710922, 14.9707870
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5617104, 13.5611420
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6641884, 14.6650963
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3071480, 13.3091164
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6544571, 16.6570053
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1540489, 14.1563873
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0096054, 13.0134048
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0993423, 15.1043053
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9085922, 18.9090271
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3612518, 18.3700638
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0774841, 16.0816956
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3445969, 14.3435326
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6897697, 16.6907196
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2691193, 17.2683945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 857

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 840

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5582088, upper bound: 12.5462047
time: 12.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5638269, upper bound: 12.5405897
time: 11.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8928032, 13.8908615
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5003586, 8.4989834
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4594917, 13.4583969
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0255852, 12.0258808
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6875687, 14.6882324
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2049026, 15.2093048
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6333618, 13.6324959
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1077499, 12.1052246
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0304680, 13.0286903
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9458084, 20.9427490
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0310783, 15.0322914
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6304131, 16.6329880
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7642670, 26.7620773
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8204803, 14.8153648
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2494316, 17.2457542
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4225788, 14.4225292
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5547791, 14.5545120
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6286011, 12.6305695
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9631767, 14.9656639
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3424835, 14.3410759
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2732239, 9.2733536
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8254395, 13.8252258
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3623886, 19.3633118
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2186623, 13.2207870
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9704437, 14.9714355
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5616531, 13.5611954
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6645317, 14.6647530
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3058090, 13.3104553
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6531296, 16.6583328
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1526527, 14.1577835
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0090561, 13.0139542
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0977554, 15.1058846
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9078598, 18.9097633
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3601837, 18.3711319
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0763016, 16.0828743
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3435402, 14.3445892
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6890373, 16.6914482
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2686234, 17.2688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 774

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 978

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5708669, upper bound: 12.5307339
time: 15.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5592938, upper bound: 12.5381816
time: 7.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8952217, 13.8951797
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4968605, 8.4951153
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4524002, 13.4521255
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0233269, 12.0219193
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6824112, 14.6824074
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2046967, 15.2035103
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6202927, 13.6175728
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0997314, 12.0998878
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0327377, 13.0318069
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9628983, 20.9639587
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0290985, 15.0271721
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6359901, 16.6318436
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7584076, 26.7562180
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8261795, 14.8257637
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2417336, 17.2388992
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4263573, 14.4289742
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5485039, 14.5499611
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6465111, 12.6455956
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9658890, 14.9662209
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3269844, 14.3295364
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2566414, 9.2584496
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8236504, 13.8234482
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3360176, 19.3412170
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2305870, 13.2322159
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9637489, 14.9647484
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5542908, 13.5542336
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6746445, 14.6731987
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3235703, 13.3217773
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6667480, 16.6679688
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1195869, 14.1230011
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9833336, 12.9857788
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0894890, 15.0915146
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8496437, 18.8572731
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3554420, 18.3553238
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.1071587, 16.1073647
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3249588, 14.3286209
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6638794, 16.6677208
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2639771, 17.2639542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 952

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5506554, upper bound: 12.5289533
time: 13.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5677417, upper bound: 12.5118027
time: 7.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8952751, 13.8951263
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4968796, 8.4950962
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4525681, 13.4519577
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0233269, 12.0219193
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6824417, 14.6823730
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2047424, 15.2034569
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6204453, 13.6174202
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0998993, 12.0997314
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0326271, 13.0319176
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9627075, 20.9641495
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0289421, 15.0273323
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6355553, 16.6322746
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7581940, 26.7564240
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8261948, 14.8257561
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2414970, 17.2391396
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4266090, 14.4287243
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5485573, 14.5499077
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6466484, 12.6454506
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9657669, 14.9663391
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3270836, 14.3294411
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2566414, 9.2584496
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8234367, 13.8236618
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3361473, 19.3410797
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2309532, 13.2318535
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9636955, 14.9648018
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5541687, 13.5543518
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6747055, 14.6731300
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3236580, 13.3216896
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6667023, 16.6680031
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1193657, 14.1232224
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9832802, 12.9858284
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0894890, 15.0915146
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8496437, 18.8572655
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3555717, 18.3551865
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.1071510, 16.1073799
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3250961, 14.3284836
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6641312, 16.6674728
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2640381, 17.2638931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 856

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5607064, upper bound: 12.5299958
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5676748, upper bound: 12.5230247
time: 14.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8906441, 13.8925323
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4986839, 8.4999084
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4573479, 13.4583778
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0253372, 12.0238838
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6861267, 14.6869888
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2076416, 15.2043762
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6310425, 13.6319466
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1044540, 12.1059341
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0291557, 13.0300751
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9429245, 20.9420013
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0323372, 15.0298767
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6293526, 16.6265564
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7627792, 26.7628403
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8151398, 14.8191223
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2470894, 17.2477837
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4220772, 14.4237823
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5530167, 14.5540543
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6301613, 12.6291618
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9640732, 14.9623909
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3410225, 14.3427467
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2703400, 9.2700920
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8249779, 13.8248253
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3619232, 19.3624649
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2190018, 13.2196655
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9707451, 14.9698029
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5608101, 13.5600319
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6647415, 14.6655807
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3090668, 13.3057213
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6533241, 16.6510849
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1546783, 14.1511688
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0114365, 13.0084572
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1018524, 15.0981293
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9048882, 18.9056396
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3685760, 18.3618011
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0751190, 16.0721664
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3396263, 14.3408184
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6898308, 16.6896935
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2676697, 17.2675667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 840

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 769

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5492826, upper bound: 12.5594447
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5480282, upper bound: 12.5606958
time: 7.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8918648, 13.8921204
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4993401, 8.4986153
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4579010, 13.4580078
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0247040, 12.0240726
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6858368, 14.6866722
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2032318, 15.2061768
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6306076, 13.6302032
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1053467, 12.1049118
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0295715, 13.0296326
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9428482, 20.9414215
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0300865, 15.0304604
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6254272, 16.6269646
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7618408, 26.7606430
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8180046, 14.8173523
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2469711, 17.2447243
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4227276, 14.4211750
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5531845, 14.5520706
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6293068, 12.6295090
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9630127, 14.9633293
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3412704, 14.3400116
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2696190, 9.2676506
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8243980, 13.8229370
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3612442, 19.3595200
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2188416, 13.2199211
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9693260, 14.9682922
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5607185, 13.5586624
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6648712, 14.6648598
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3029099, 13.3075562
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6478271, 16.6544418
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1521072, 14.1551056
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0107727, 13.0123482
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0997467, 15.1037598
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9032707, 18.9068985
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3625412, 18.3660278
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0660782, 16.0731049
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3345337, 14.3398933
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6876678, 16.6914024
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2653961, 17.2684364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 890

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 972

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5585700, upper bound: 12.5471750
time: 7.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5606582, upper bound: 12.5450784
time: 6.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9002724, 13.9002151
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5005608, 8.5022125
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4603043, 13.4599380
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0359840, 12.0359840
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6806030, 14.6816788
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2079010, 15.2074280
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6426010, 13.6439743
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1267509, 12.1290169
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0519562, 13.0501328
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9501266, 20.9495392
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0208855, 15.0198593
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6491928, 16.6440697
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7639847, 26.7596207
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8234062, 14.8247108
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2372398, 17.2362747
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4322281, 14.4331722
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5479279, 14.5483246
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6234436, 12.6227493
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9666824, 14.9651947
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3294640, 14.3289452
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2753372, 9.2747116
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8087997, 13.8082733
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3219070, 19.3216476
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2163353, 13.2171860
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9476547, 14.9439583
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5600357, 13.5591698
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6638870, 14.6657104
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3211212, 13.3215027
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6014709, 16.6085854
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1359863, 14.1448822
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0012283, 13.0054893
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0988007, 15.0980988
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9229088, 18.9243622
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3540421, 18.3577499
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0479584, 16.0515747
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3727341, 14.3779984
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7112274, 16.7151337
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2996178, 17.3014374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 956

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5535963, upper bound: 12.5092245
time: 22.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5618113, upper bound: 12.5009992
time: 29.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9101372, 13.9091148
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5087395, 8.5076065
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4592514, 13.4591255
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0356789, 12.0348854
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6813507, 14.6809311
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1916389, 15.1891365
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6385040, 13.6376266
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1112480, 12.1124973
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0441437, 13.0428314
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9522705, 20.9524002
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0135994, 15.0110550
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6049500, 16.6019096
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7641220, 26.7647171
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8246841, 14.8229218
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2267685, 17.2269516
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4305000, 14.4299603
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5516739, 14.5516281
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6343613, 12.6350517
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9613113, 14.9623566
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3245964, 14.3254471
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2743301, 9.2761497
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8093872, 13.8117027
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3676453, 19.3698578
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2069244, 13.2092209
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9622803, 14.9639206
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5544243, 13.5552750
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6565857, 14.6576462
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3076477, 13.3054581
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6428070, 16.6402779
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1445312, 14.1422119
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0034561, 13.0016594
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1169052, 15.1173744
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9231110, 18.9224701
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3671112, 18.3661499
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0571098, 16.0558128
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3464813, 14.3461838
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6951904, 16.6930428
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2750740, 17.2754555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 978

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 948

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5417264, upper bound: 12.5671841
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5455192, upper bound: 12.5634125
time: 6.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9097214, 13.9095306
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5084839, 8.5078621
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4595413, 13.4588394
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0356026, 12.0349579
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6817627, 14.6805191
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1916008, 15.1891708
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6385193, 13.6376076
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1125298, 12.1112156
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0438385, 13.0431366
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9519043, 20.9527664
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0121574, 15.0124969
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6024170, 16.6044464
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7632980, 26.7655411
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8241653, 14.8234406
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2252426, 17.2284698
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4305229, 14.4299355
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5518265, 14.5514755
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6349945, 12.6344185
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9617691, 14.9618988
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3247643, 14.3252754
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2743607, 9.2761192
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8093796, 13.8117142
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3682022, 19.3693008
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2089615, 13.2071838
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9622955, 14.9639053
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5539665, 13.5557327
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6573868, 14.6568451
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3077965, 13.3053055
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6428986, 16.6401863
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1447525, 14.1419868
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0034332, 13.0016823
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1170120, 15.1172638
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9231186, 18.9224663
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3671265, 18.3661346
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0571098, 16.0558128
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3469276, 14.3457375
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6953201, 16.6929169
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2755470, 17.2749825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 921

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 899

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5375563, upper bound: 12.5677814
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5450793, upper bound: 12.5602592
time: 6.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8975220, 13.8974838
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5021477, 8.5013885
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4610443, 13.4602089
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0306244, 12.0318375
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6853561, 14.6852913
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2163086, 15.2179756
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6437073, 13.6430779
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1293411, 12.1247845
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0503502, 13.0529709
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9468842, 20.9470825
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0224228, 15.0263214
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6392899, 16.6496506
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7696228, 26.7737656
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8226357, 14.8213005
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2446785, 17.2464867
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4316921, 14.4292183
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5485420, 14.5475731
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6232605, 12.6230316
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9673615, 14.9693069
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3352890, 14.3332825
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2681236, 9.2668571
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8152962, 13.8145447
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3290176, 19.3249435
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2200050, 13.2164116
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9547386, 14.9568176
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5623627, 13.5628357
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6658287, 14.6637573
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3296776, 13.3320427
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6278534, 16.6257057
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1626053, 14.1569939
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0168991, 13.0169487
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0964279, 15.0992012
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9251022, 18.9244003
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3698769, 18.3695984
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0629234, 16.0644875
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3821373, 14.3761044
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7205162, 16.7171402
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2992172, 17.2964134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 771

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 946

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5190403, upper bound: 12.5620950
time: 38.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5141878, upper bound: 12.5675293
time: 7.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8966599, 13.8983421
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5008545, 8.5026817
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4608765, 13.4603767
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0317917, 12.0306740
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6851425, 14.6855049
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2174454, 15.2168388
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6430969, 13.6436920
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1269455, 12.1271763
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0510254, 13.0522957
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9470673, 20.9469147
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0236244, 15.0251160
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6438828, 16.6450615
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7726593, 26.7707214
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8201790, 14.8237572
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2449608, 17.2462082
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4305325, 14.4303741
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5479774, 14.5481377
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6240730, 12.6222229
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9694595, 14.9672089
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3350372, 14.3335342
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2678299, 9.2671509
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8149223, 13.8149185
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3264160, 19.3275452
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2196045, 13.2168121
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9573097, 14.9542503
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5624886, 13.5627098
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6651421, 14.6644440
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3318100, 13.3299103
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6239090, 16.6296501
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1567764, 14.1628265
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0157700, 13.0180817
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0983887, 15.0972443
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9231262, 18.9263725
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3688316, 18.3706360
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0617638, 16.0656471
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3763542, 14.3818855
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7176170, 16.7200356
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2970428, 17.2985878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 771

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 768

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5613010, upper bound: 12.5247645
time: 19.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5607496, upper bound: 12.5253211
time: 7.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9045677, 13.9053764
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5079384, 8.5090370
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4613647, 13.4606018
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0310745, 12.0316448
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6869354, 14.6863174
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2049141, 15.2060471
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6374207, 13.6386414
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1156921, 12.1139221
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0455284, 13.0455666
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9497681, 20.9489594
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0246658, 15.0269279
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6068687, 16.6110001
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7721558, 26.7736053
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8248520, 14.8256874
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2451553, 17.2470360
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4289951, 14.4283752
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5527725, 14.5519333
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6368637, 12.6364708
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9666710, 14.9663811
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3330231, 14.3309326
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2708969, 9.2701073
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8186836, 13.8186531
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3716278, 19.3698807
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2148743, 13.2130013
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9733238, 14.9730530
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5611534, 13.5620308
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6597214, 14.6595039
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3147278, 13.3148994
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6641159, 16.6638107
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1622810, 14.1616020
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0165024, 13.0163345
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1159821, 15.1158333
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9240112, 18.9235229
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3803940, 18.3811493
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0693855, 16.0706367
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3469963, 14.3456230
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6986084, 16.6978264
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2699280, 17.2690392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 779

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1574

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5427497, upper bound: 12.5563017
time: 9.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5381349, upper bound: 12.5609085
time: 7.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9021034, 13.9014969
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5013275, 8.5017643
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4580460, 13.4576607
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0204849, 12.0216599
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6919708, 14.6908417
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9076614, 15.9097824
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2123795, 15.2124557
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6215515, 13.6227455
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1138077, 12.1136112
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0510674, 13.0506554
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9437714, 20.9440918
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0248566, 15.0257339
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6180115, 16.6185799
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7567139, 26.7594604
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8242760, 14.8236961
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2342834, 17.2382317
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4315834, 14.4303303
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5519295, 14.5508232
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6371765, 12.6380348
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9623566, 14.9619102
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3335266, 14.3325958
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2766228, 9.2757111
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8317680, 13.8334694
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3604050, 19.3582306
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2134285, 13.2121391
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9701614, 14.9708252
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5629578, 13.5643539
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6621132, 14.6625290
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3244858, 13.3235054
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6146622, 16.6072884
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1498260, 14.1448097
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9943542, 12.9896126
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1155014, 15.1123810
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8515587, 18.8429489
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3733940, 18.3710175
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0488548, 16.0419121
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3091278, 14.3023472
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6800690, 16.6747704
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2655792, 17.2647285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 782

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1294

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5219399, upper bound: 12.5718214
time: 9.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5223592, upper bound: 12.5714039
time: 7.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9008102, 13.9013748
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4996605, 8.4997635
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4584160, 13.4580917
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0238190, 12.0242786
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6872482, 14.6861000
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9152832, 15.9154549
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2097054, 15.2101898
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6279144, 13.6277313
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1066170, 12.1069336
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0491638, 13.0502777
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9465103, 20.9482040
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0161018, 15.0176125
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6344337, 16.6339455
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7676010, 26.7679367
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8214722, 14.8231621
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2491989, 17.2506561
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4336052, 14.4321556
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5560608, 14.5549583
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6393776, 12.6380043
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9630508, 14.9618721
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3372459, 14.3372498
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2870979, 9.2857666
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8351250, 13.8338928
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3648338, 19.3636017
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2132912, 13.2110825
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9713936, 14.9710922
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5607681, 13.5609360
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6638832, 14.6618919
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3190804, 13.3196945
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6204605, 16.6201401
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1640244, 14.1646957
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0048141, 13.0037575
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1105881, 15.1083221
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8678513, 18.8690529
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3714218, 18.3687668
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0428505, 16.0408440
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3182716, 14.3205948
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6878586, 16.6888008
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2627716, 17.2644272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 947

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 988

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5476754, upper bound: 12.5643035
time: 20.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5478023, upper bound: 12.5641768
time: 7.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9009590, 13.9012222
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4992790, 8.5001450
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4584236, 13.4580803
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0238113, 12.0242863
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6868210, 14.6865273
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9152527, 15.9154930
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2100487, 15.2098427
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6275787, 13.6280670
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1065254, 12.1070251
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0494843, 13.0499573
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9465790, 20.9481430
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0165749, 15.0171394
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6350288, 16.6333427
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7675400, 26.7680054
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8213501, 14.8232841
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2486954, 17.2511597
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4332390, 14.4325256
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5556793, 14.5553322
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6393471, 12.6380348
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9630432, 14.9618835
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3373222, 14.3371658
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2865181, 9.2863464
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8346291, 13.8343887
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3651772, 19.3632584
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2131958, 13.2111778
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9713783, 14.9711075
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5602417, 13.5614586
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6632500, 14.6625252
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3198318, 13.3189468
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6210556, 16.6195450
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1643982, 14.1643219
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0050735, 13.0034981
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1113510, 15.1075516
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8688736, 18.8680267
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3720169, 18.3681717
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0434608, 16.0402298
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3190536, 14.3198128
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6886444, 16.6880150
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2636719, 17.2635269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 775

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5453301, upper bound: 12.5631840
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5365538, upper bound: 12.5668200
time: 9.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8957481, 13.8968430
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5020523, 8.5029068
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4559250, 13.4563904
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0260353, 12.0259628
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6869965, 14.6868172
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2132874, 15.2112885
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6272354, 13.6281815
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1065025, 12.1065102
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0413971, 13.0419426
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9510040, 20.9518585
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0257797, 15.0246468
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6277122, 16.6285286
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7726822, 26.7736588
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8168106, 14.8178864
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2520866, 17.2537727
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4321308, 14.4323196
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5534325, 14.5538559
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6350822, 12.6343460
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9605751, 14.9599953
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3365860, 14.3369217
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2757607, 9.2767181
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8235931, 13.8240471
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3603745, 19.3590622
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2220116, 13.2217522
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9686584, 14.9677963
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5587654, 13.5588989
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6598969, 14.6606255
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3213539, 13.3193779
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6474762, 16.6460915
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1636696, 14.1612816
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0080223, 13.0071182
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1142731, 15.1129723
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9055099, 18.9046478
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3752289, 18.3730316
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0569000, 16.0561981
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3369255, 14.3368301
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7008820, 16.6991310
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2764626, 17.2755890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 887

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 889

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5248139, upper bound: 12.5593526
time: 16.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5401412, upper bound: 12.5440260
time: 7.97 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 26.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5704547, upper bound: 12.5329799
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5688353, upper bound: 12.5346068
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5582088, upper bound: 12.5462047
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5638269, upper bound: 12.5405897
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5708669, upper bound: 12.5307339
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5592938, upper bound: 12.5381816
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5506554, upper bound: 12.5289533
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5677417, upper bound: 12.5118027
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5607064, upper bound: 12.5299958
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5676748, upper bound: 12.5230247
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5492826, upper bound: 12.5594447
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5480282, upper bound: 12.5606958
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5585700, upper bound: 12.5471750
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5606582, upper bound: 12.5450784
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5535963, upper bound: 12.5092245
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5618113, upper bound: 12.5009992
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5417264, upper bound: 12.5671841
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5455192, upper bound: 12.5634125
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5375563, upper bound: 12.5677814
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5450793, upper bound: 12.5602592
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5190403, upper bound: 12.5620950
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5141878, upper bound: 12.5675293
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5613010, upper bound: 12.5247645
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5607496, upper bound: 12.5253211
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5427497, upper bound: 12.5563017
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5381349, upper bound: 12.5609085
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5219399, upper bound: 12.5718214
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5223592, upper bound: 12.5714039
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5476754, upper bound: 12.5643035
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5478023, upper bound: 12.5641768
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5453301, upper bound: 12.5631840
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5365538, upper bound: 12.5668200
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5248139, upper bound: 12.5593526
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 26.36
Output dim: 14, lower bound: -12.5401412, upper bound: 12.5440260

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8954849, 13.8941841
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5060463, 8.5054111
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4598083, 13.4587288
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0233383, 12.0233459
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6911163, 14.6923447
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2067947, 15.2105980
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6339951, 13.6331329
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1028709, 12.1002350
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0342827, 13.0331192
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9396362, 20.9361038
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0303078, 15.0319519
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6211586, 16.6258049
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7647247, 26.7623138
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8230362, 14.8190613
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2353821, 17.2302132
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4274788, 14.4277573
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5484657, 14.5487556
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6296730, 12.6313210
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9629593, 14.9652557
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3399734, 14.3384972
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2711754, 9.2712250
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8236847, 13.8237000
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3601341, 19.3612900
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2206383, 13.2224503
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9747772, 14.9751167
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5615921, 13.5603676
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6597710, 14.6603889
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3065872, 13.3111534
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6399918, 16.6470566
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1510925, 14.1565475
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9919968, 12.9990273
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0833588, 15.0932770
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8946304, 18.8985062
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3410416, 18.3542709
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0498161, 16.0599136
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3494949, 14.3497314
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6851997, 16.6883049
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2744064, 17.2741013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1292

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 946

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5704211, upper bound: 12.5280914
time: 11.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5655904, upper bound: 12.5329461
time: 36.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8956566, 13.8940086
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5060081, 8.5054531
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4598083, 13.4587288
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0233459, 12.0233383
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6909485, 14.6925163
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2068024, 15.2105980
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6338882, 13.6332397
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1028481, 12.1002579
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0343781, 13.0330238
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9396286, 20.9361038
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0303535, 15.0319061
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6215782, 16.6253929
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7646790, 26.7623596
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8229980, 14.8190994
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2350235, 17.2305679
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4274025, 14.4278336
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5484047, 14.5488052
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6295586, 12.6314354
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9628983, 14.9653206
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3401642, 14.3383141
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2711334, 9.2712631
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8236465, 13.8237419
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3602791, 19.3611450
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2206192, 13.2224693
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9746780, 14.9752159
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5612640, 13.5606956
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6596870, 14.6604729
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3067665, 13.3109741
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6404114, 16.6466331
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1512756, 14.1563683
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9922485, 12.9987717
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0838547, 15.0927887
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8952866, 18.8978348
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3416138, 18.3536987
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0504951, 16.0592346
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3498306, 14.3493958
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6855507, 16.6879539
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2748032, 17.2737007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 978

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5683608, upper bound: 12.5293379
time: 8.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5569171, upper bound: 12.5339121
time: 8.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8927727, 13.8927002
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5000954, 8.4996014
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4645920, 13.4633026
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0173874, 12.0191269
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6901169, 14.6893387
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1960030, 15.1996078
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6388550, 13.6380920
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1125412, 12.1098003
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0371971, 13.0378418
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9403229, 20.9388046
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0180550, 15.0215645
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5976181, 16.6041565
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7658539, 26.7654114
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8257637, 14.8229408
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2497711, 17.2468414
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4028702, 14.4009895
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5418777, 14.5406914
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6303406, 12.6307716
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9663239, 14.9670792
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3347549, 14.3335190
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2586708, 9.2563705
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8205452, 13.8202209
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3526115, 19.3517609
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2048531, 13.2040443
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9697037, 14.9692268
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5610352, 13.5595741
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6519470, 14.6510887
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2831459, 13.2881317
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6493645, 16.6557236
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1461983, 14.1514740
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0191231, 13.0251694
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0995636, 15.1072922
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9162521, 18.9183197
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3499985, 18.3610229
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0492706, 16.0571671
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3331528, 14.3334846
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6838074, 16.6857605
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2569199, 17.2583313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 950

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5350161, upper bound: 12.5400337
time: 18.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5632719, upper bound: 12.5117545
time: 7.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8922882, 13.8901253
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4998245, 8.4982433
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4594650, 13.4583473
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0250397, 12.0251637
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6873856, 14.6883698
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2041245, 15.2086411
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6330032, 13.6319618
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1068878, 12.1039886
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0298767, 13.0279388
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9441071, 20.9403839
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0308685, 15.0320892
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6298714, 16.6331367
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7628937, 26.7601395
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8182945, 14.8125153
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2463112, 17.2416382
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4220791, 14.4221001
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5540695, 14.5539665
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6273003, 12.6296043
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9615288, 14.9644623
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3425217, 14.3410301
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2730904, 9.2732582
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8252068, 13.8251114
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3607788, 19.3620911
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2170219, 13.2195587
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9703293, 14.9713478
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5612946, 13.5605927
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6640205, 14.6643372
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3031693, 13.3084030
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6494598, 16.6556816
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1504173, 14.1562233
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0053101, 13.0112648
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0918045, 15.1015129
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9043274, 18.9070930
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3528442, 18.3658066
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0698547, 16.0782013
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3429794, 14.3439789
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6864433, 16.6895065
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2686920, 17.2687798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1284

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 952

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5341931, upper bound: 12.5292782
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5694125, upper bound: 12.4940327
time: 8.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8957596, 13.8957596
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4966545, 8.4949303
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4524460, 13.4521561
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0164146, 12.0141716
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6880341, 14.6886902
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9155579, 15.9142494
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2033043, 15.2022438
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6136398, 13.6103020
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1049232, 12.1056538
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0324898, 13.0315323
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9617920, 20.9629364
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0334206, 15.0310440
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6283913, 16.6226730
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7504196, 26.7468185
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8252411, 14.8253403
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2309799, 17.2264595
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4250431, 14.4281921
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5449944, 14.5473137
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6455612, 12.6443214
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9656219, 14.9658585
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3247032, 14.3276367
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2470665, 9.2497597
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8248596, 13.8246078
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3319244, 19.3382645
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2339134, 13.2356644
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9635620, 14.9645348
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5542641, 13.5542107
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6746025, 14.6731644
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3207054, 13.3186073
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6467781, 16.6497231
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1143227, 14.1194115
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9704628, 12.9742470
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0907936, 15.0928574
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8153572, 18.8260078
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3552132, 18.3552094
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0989494, 16.1004524
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3100739, 14.3164024
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6541443, 16.6596489
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2640533, 17.2640686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1644

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 904

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5284671, upper bound: 12.4850242
time: 30.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5404283, upper bound: 12.4732533
time: 10.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8957253, 13.8950996
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4964523, 8.4948254
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4525223, 13.4520531
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0229340, 12.0211067
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6820602, 14.6821251
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2033539, 15.2000504
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6191177, 13.6165085
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0994568, 12.0996151
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0327263, 13.0313339
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9626083, 20.9633255
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0278740, 15.0252342
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6335945, 16.6284561
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7566833, 26.7543411
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8261108, 14.8257103
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2388039, 17.2378311
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4245300, 14.4281006
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5469933, 14.5491524
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6460266, 12.6450882
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9656754, 14.9662514
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3249245, 14.3282738
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2541313, 9.2575798
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8218117, 13.8228226
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3331032, 19.3398209
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2307205, 13.2318382
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9615364, 14.9637642
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5527420, 13.5542641
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6743126, 14.6729202
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3220406, 13.3168716
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6663589, 16.6638374
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1199837, 14.1220474
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9847679, 12.9857178
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0915833, 15.0909882
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8499680, 18.8563843
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3538132, 18.3502274
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.1033401, 16.0995178
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3231087, 14.3244133
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6642761, 16.6660347
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2639771, 17.2621040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1764

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 890

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5490523, upper bound: 12.5296854
time: 7.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5603954, upper bound: 12.5183289
time: 13.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8952484, 13.8955765
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4966164, 8.4946575
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4526672, 13.4519157
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0225143, 12.0215263
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6821899, 14.6819954
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2013474, 15.2020645
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6195297, 13.6160965
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0997772, 12.0992908
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0320473, 13.0320129
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9618759, 20.9640579
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0268440, 15.0262642
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6317406, 16.6303101
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7561188, 26.7549057
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8261490, 14.8256760
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2401848, 17.2364540
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4259872, 14.4266434
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5478020, 14.5483437
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6462936, 12.6448288
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9656677, 14.9662628
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3259163, 14.3272820
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2557716, 9.2559395
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8225975, 13.8220329
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3348885, 19.3380356
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2309380, 13.2316208
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9626579, 14.9626389
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5540810, 13.5529251
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6744881, 14.6727409
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3188362, 13.3200722
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6625443, 16.6676483
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1181908, 14.1238403
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9831734, 12.9873161
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0889664, 15.0936127
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8487701, 18.8575783
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3506088, 18.3534317
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0992889, 16.1035652
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3210220, 14.3264999
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6626968, 16.6676178
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2622452, 17.2638321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 938

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 689

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5673733, upper bound: 12.5061928
time: 7.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5508465, upper bound: 12.5227227
time: 6.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8906021, 13.8922348
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4987640, 8.4998512
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4572258, 13.4580116
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0253067, 12.0238647
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6861267, 14.6867561
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2069855, 15.2035637
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6311874, 13.6317635
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1043777, 12.1056900
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0290375, 13.0299988
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9414062, 20.9409866
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0322609, 15.0299797
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6282272, 16.6256943
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7614288, 26.7618484
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8151932, 14.8190689
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2458153, 17.2471313
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4215508, 14.4229660
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5529251, 14.5538635
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6300926, 12.6289825
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9637299, 14.9622917
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3407326, 14.3423729
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2701836, 9.2699318
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8237457, 13.8240814
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3619156, 19.3624496
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2177811, 13.2180405
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9697571, 14.9691544
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5589638, 13.5587044
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6648254, 14.6655159
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3084602, 13.3046989
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6529503, 16.6506500
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1545677, 14.1510315
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0113945, 13.0083923
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1011810, 15.0972481
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9041328, 18.9047050
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3669891, 18.3595657
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0741043, 16.0709381
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3392525, 14.3401756
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6885567, 16.6878929
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2673111, 17.2671585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 976

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5479836, upper bound: 12.5560190
time: 9.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5433464, upper bound: 12.5606515
time: 16.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8918114, 13.8927078
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4993362, 8.4986153
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4578934, 13.4577980
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0249329, 12.0239716
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6852493, 14.6862717
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2027893, 15.2058182
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6308441, 13.6301727
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1065674, 12.1047935
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0294342, 13.0311127
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9426651, 20.9412689
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0263634, 15.0285378
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6197472, 16.6245079
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7615204, 26.7602844
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8166962, 14.8167191
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2461357, 17.2438583
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4212208, 14.4190331
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5530930, 14.5519295
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6297798, 12.6291656
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9626160, 14.9628677
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3396912, 14.3376007
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2683258, 9.2653351
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8211632, 13.8183403
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3611679, 19.3592606
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2148666, 13.2139549
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9684563, 14.9670792
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5608635, 13.5582771
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6646233, 14.6633682
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3003502, 13.3057175
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6419144, 16.6507225
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1518250, 14.1551552
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0094376, 13.0129013
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0977364, 15.1028252
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9018631, 18.9081306
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3618813, 18.3655243
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0562019, 16.0659790
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3315468, 14.3377495
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6851425, 16.6897278
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2652283, 17.2677994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1285

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1288

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5599513, upper bound: 12.5449543
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5605347, upper bound: 12.5443683
time: 7.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9026489, 13.9029236
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4988136, 8.5002098
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4598923, 13.4590912
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0357132, 12.0356941
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6809769, 14.6820335
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1896210, 15.1909828
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6423492, 13.6436920
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1416054, 12.1420403
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0557404, 13.0544510
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9511948, 20.9506226
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0063210, 15.0076599
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6572113, 16.6552734
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7635651, 26.7591705
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8226585, 14.8239670
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2277527, 17.2274475
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4263458, 14.4264507
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5528221, 14.5524330
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6280785, 12.6266747
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9642868, 14.9625702
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3263779, 14.3246574
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2727089, 9.2708740
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7981224, 13.7959480
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3313675, 19.3299561
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2217102, 13.2205276
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9473228, 14.9435234
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5575333, 13.5563812
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6725464, 14.6725960
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3078270, 13.3098640
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5904999, 16.5994873
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1288452, 14.1389236
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9943581, 12.9997253
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0907135, 15.0910645
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9230461, 18.9263191
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3487320, 18.3529510
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0346603, 16.0399361
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3635139, 14.3699265
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7008934, 16.7060928
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2976875, 17.2995987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 772

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5441861, upper bound: 12.5004343
time: 25.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5612490, upper bound: 12.4833679
time: 29.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9151573, 13.9134636
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5106049, 8.5094795
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4572792, 13.4567986
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0319977, 12.0319939
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6766968, 14.6751633
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1919098, 15.1899261
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6364059, 13.6354332
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1130066, 12.1146469
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0457191, 13.0441628
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9530640, 20.9546967
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0134201, 15.0108376
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6077995, 16.6038208
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7570343, 26.7593079
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8267670, 14.8247871
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2313347, 17.2323952
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4304466, 14.4299431
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5496216, 14.5495453
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6364632, 12.6375999
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9583511, 14.9596443
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3243484, 14.3256683
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2745743, 9.2764359
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8111877, 13.8152161
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3734894, 19.3763809
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2089844, 13.2116661
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9546967, 14.9573288
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5582886, 13.5599022
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6571121, 14.6583023
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3156815, 13.3117981
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6211472, 16.6151810
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1514511, 14.1482315
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0098076, 13.0066071
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1243362, 15.1234283
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9056969, 18.9017525
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3676147, 18.3660126
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0290031, 16.0237350
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3289146, 14.3255386
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7028046, 16.6985016
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2762146, 17.2765465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1291

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5406509, upper bound: 12.5670388
time: 9.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5415800, upper bound: 12.5661128
time: 15.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9144897, 13.9141350
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5106125, 8.5094719
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4569283, 13.4571495
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0327835, 12.0312042
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6755981, 14.6762657
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1924286, 15.1894035
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6363068, 13.6355286
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1133957, 12.1142540
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0454712, 13.0444069
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9545670, 20.9531937
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0133820, 15.0108757
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6068611, 16.6047516
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7587128, 26.7576141
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8265457, 14.8250084
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2322121, 17.2315216
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4304810, 14.4299088
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5495911, 14.5495796
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6369057, 12.6371574
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9586029, 14.9593964
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3248138, 14.3251991
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2746124, 9.2763939
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8129044, 13.8134956
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3741760, 19.3756943
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2093697, 13.2112846
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9556885, 14.9563370
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5590515, 13.5591431
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6572418, 14.6581764
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3139877, 13.3134918
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6177139, 16.6186142
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1505432, 14.1491356
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0084038, 13.0080109
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1229553, 15.1248016
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9024010, 18.9050484
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3669739, 18.3666534
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0250282, 16.0277100
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3258362, 14.3286190
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7006454, 16.7006569
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2761688, 17.2765923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1284

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5447459, upper bound: 12.5632768
time: 6.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5453838, upper bound: 12.5626388
time: 7.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9033852, 13.9040565
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5013123, 8.5015526
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4568901, 13.4565163
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0340080, 12.0335350
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6804276, 14.6791000
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1849327, 15.1814957
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6351852, 13.6347351
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1103897, 12.1096573
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0396271, 13.0394669
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9478760, 20.9492188
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0089340, 15.0091057
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6062431, 16.6081772
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7570343, 26.7599640
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8140488, 14.8150215
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2271538, 17.2307549
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4248219, 14.4247513
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5548859, 14.5549583
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6316338, 12.6305580
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9581528, 14.9573174
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3280678, 14.3291473
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2753792, 9.2770691
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8111038, 13.8137016
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3673630, 19.3683548
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2073250, 13.2050591
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9585953, 14.9595490
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5531883, 13.5548973
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6604233, 14.6602325
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2985382, 13.2947083
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6323395, 16.6282959
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1351433, 14.1309776
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9994431, 12.9971466
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1079941, 15.1066628
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9179840, 18.9165916
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3580933, 18.3560257
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0494080, 16.0469284
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3398552, 14.3376083
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6900101, 16.6868820
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2713127, 17.2702484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 782

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1284

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5367829, upper bound: 12.5676464
time: 7.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5374208, upper bound: 12.5670108
time: 19.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8968277, 13.8966484
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5002823, 8.5000114
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4607086, 13.4598961
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0302734, 12.0322132
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6843643, 14.6844482
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2153931, 15.2171822
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6432266, 13.6429405
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1286240, 12.1247749
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0502396, 13.0527992
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9464417, 20.9466476
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0224915, 15.0262833
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6346931, 16.6435623
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7678528, 26.7718887
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8209724, 14.8200989
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2425919, 17.2436218
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4308758, 14.4286098
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5481148, 14.5473213
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6230927, 12.6229324
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9664345, 14.9682617
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3350830, 14.3330650
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2676582, 9.2662430
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8142624, 13.8136101
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3255310, 19.3213348
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2196236, 13.2160988
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9528961, 14.9545937
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5609131, 13.5612106
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6637497, 14.6623840
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3282166, 13.3307076
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6255417, 16.6234970
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1589088, 14.1540222
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0150070, 13.0155334
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0952530, 15.0978661
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9250298, 18.9241600
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3697929, 18.3695221
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0627861, 16.0643234
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3812752, 14.3750114
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7190437, 16.7157669
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2981262, 17.2951813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1284

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 874

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5120873, upper bound: 12.5625793
time: 7.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5189382, upper bound: 12.5557270
time: 6.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8966866, 13.8967896
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5007668, 8.4995270
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4607315, 13.4598770
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0309982, 12.0314884
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6845093, 14.6842995
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2155151, 15.2170601
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6435699, 13.6425972
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1293335, 12.1240692
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0501823, 13.0528603
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9464493, 20.9466400
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0223846, 15.0263901
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6332054, 16.6450500
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7677307, 26.7720032
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8214378, 14.8196373
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2418213, 17.2444000
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4310818, 14.4284019
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5482903, 14.5471458
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6231613, 12.6228638
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9663124, 14.9683800
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3350754, 14.3330765
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2675095, 9.2663918
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8143539, 13.8135033
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3254013, 19.3214569
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2196960, 13.2160263
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9525146, 14.9549713
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5607414, 13.5613823
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6644516, 14.6616783
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3283424, 13.3305817
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6256485, 16.6233902
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1596336, 14.1532974
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0154953, 13.0150490
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0950928, 15.0980225
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9248619, 18.9243317
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3698082, 18.3695145
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0627556, 16.0643463
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3810425, 14.3752441
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7191429, 16.7156677
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2979813, 17.2953224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 948

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5102173, upper bound: 12.5673512
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5140085, upper bound: 12.5635610
time: 7.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8965645, 13.8983002
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5008392, 8.5026817
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4607315, 13.4603996
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0317535, 12.0306358
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6850662, 14.6854630
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2172852, 15.2167320
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6429596, 13.6437073
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1268387, 12.1272297
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0510674, 13.0522270
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9470215, 20.9466705
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0236053, 15.0249405
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6435471, 16.6442986
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7724304, 26.7702866
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8201714, 14.8237610
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2447319, 17.2457390
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4301662, 14.4302597
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5479622, 14.5481796
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6240540, 12.6223488
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9694786, 14.9671097
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3348770, 14.3334656
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2678108, 9.2671318
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8148117, 13.8145905
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3261032, 19.3273697
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2189255, 13.2164955
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9572220, 14.9541168
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5621758, 13.5622787
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6651306, 14.6644974
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3316841, 13.3298759
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6237411, 16.6294479
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1569748, 14.1628075
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0157166, 13.0179825
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0983887, 15.0972443
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9230652, 18.9263229
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3685646, 18.3704987
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0616837, 16.0655518
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3762665, 14.3819389
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7170906, 16.7197571
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2968903, 17.2985001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 955

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5419198, upper bound: 12.5246487
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5611783, upper bound: 12.5053780
time: 14.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8966179, 13.8982468
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5008583, 8.5026627
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4608994, 13.4602318
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0317459, 12.0306396
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6850967, 14.6854324
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2173309, 15.2166748
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6431122, 13.6435547
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1269989, 12.1270733
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0509567, 13.0523376
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9468231, 20.9468689
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0234451, 15.0251007
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6431198, 16.6447296
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7722168, 26.7704926
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8201790, 14.8237534
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2444878, 17.2459793
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4304180, 14.4300079
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5480232, 14.5481262
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6241989, 12.6222038
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9693642, 14.9672279
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3349686, 14.3333702
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2678108, 9.2671318
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8145981, 13.8148041
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3262405, 19.3272400
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2192879, 13.2161331
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9571686, 14.9541702
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5620575, 13.5624008
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6651993, 14.6644325
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3317719, 13.3297844
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6237106, 16.6294785
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1567535, 14.1630287
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0156708, 13.0180321
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0983887, 15.0972443
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9230804, 18.9263077
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3686943, 18.3703613
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0616684, 16.0655670
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3764076, 14.3817997
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7173424, 16.7195091
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2969513, 17.2984390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1785

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1283

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5598685, upper bound: 12.5251621
time: 6.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5605904, upper bound: 12.5244396
time: 14.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9044914, 13.9053268
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5062714, 8.5083809
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4612122, 13.4607315
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0297089, 12.0293922
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6859589, 14.6866531
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2029228, 15.2022743
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6373901, 13.6388741
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1147537, 12.1143074
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0455246, 13.0455475
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9485931, 20.9472122
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0230675, 15.0232201
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6050148, 16.6070137
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7722168, 26.7735901
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8231773, 14.8254509
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2451630, 17.2469635
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4276237, 14.4278088
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5497894, 14.5502167
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6368866, 12.6364517
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9664154, 14.9660225
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3321075, 14.3304062
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2669106, 9.2680817
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8145714, 13.8158951
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3709793, 19.3691635
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2122536, 13.2115555
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9735374, 14.9729729
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5601730, 13.5613174
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6563148, 14.6578751
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3130035, 13.3113708
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6638412, 16.6635017
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1611328, 14.1600838
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0163803, 13.0160408
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1158981, 15.1144066
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9237213, 18.9232483
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3796768, 18.3790054
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0692444, 16.0705109
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3469162, 14.3455505
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6972122, 16.6958351
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2686768, 17.2670059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 979

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5378856, upper bound: 12.5492224
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5264479, upper bound: 12.5606579
time: 12.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9022141, 13.9015732
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5014877, 8.5018997
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4582977, 13.4579124
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0205154, 12.0216942
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6920395, 14.6908989
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9079742, 15.9100838
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2122269, 15.2122917
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6219940, 13.6231575
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1139374, 12.1137657
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0505905, 13.0501862
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9428558, 20.9432373
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0246620, 15.0255470
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6178589, 16.6184082
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7567215, 26.7594681
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8242264, 14.8236542
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2342911, 17.2382584
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4310665, 14.4297523
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5516968, 14.5505486
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6369095, 12.6377716
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9625969, 14.9622002
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3327713, 14.3317909
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2761803, 9.2752457
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8317795, 13.8334732
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3603630, 19.3581848
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2134743, 13.2121773
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9703674, 14.9710655
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5626678, 13.5641022
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6615639, 14.6619759
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3245888, 13.3235931
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6155052, 16.6080475
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1505241, 14.1455383
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9950218, 12.9902153
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1155319, 15.1124115
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8520393, 18.8433838
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3724365, 18.3699799
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0490189, 16.0420570
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3083420, 14.3015060
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6801720, 16.6748581
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2657013, 17.2648315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1594

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 977

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5212867, upper bound: 12.5597005
time: 9.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5119926, upper bound: 12.5712826
time: 7.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9021797, 13.9016075
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5014610, 8.5019264
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4582977, 13.4579124
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0205154, 12.0216942
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6920395, 14.6909065
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9079666, 15.9100914
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2122116, 15.2122993
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6219635, 13.6231956
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1139603, 12.1137428
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0505943, 13.0501785
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9429245, 20.9431763
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0246735, 15.0255356
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6178436, 16.6184235
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7567215, 26.7594604
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8242340, 14.8236504
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2343140, 17.2382355
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4310055, 14.4298115
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5516510, 14.5505867
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6369057, 12.6377716
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9626427, 14.9621620
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3327255, 14.3318367
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2761574, 9.2752724
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8317795, 13.8334732
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3603554, 19.3581848
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2134628, 13.2121887
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9704056, 14.9710312
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5627060, 13.5640640
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6615562, 14.6619835
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3245697, 13.3236122
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6154289, 16.6081314
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1505547, 14.1455040
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9949532, 12.9902840
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1155243, 15.1124115
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8519936, 18.8434296
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3723602, 18.3700562
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0489960, 16.0420761
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3082848, 14.3015652
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6801567, 16.6748695
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2656784, 17.2648506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 952

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 769

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5222689, upper bound: 12.5700741
time: 8.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5210145, upper bound: 12.5713117
time: 6.31 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 16.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5704211, upper bound: 12.5280914
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5655904, upper bound: 12.5329461
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5683608, upper bound: 12.5293379
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5569171, upper bound: 12.5339121
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5350161, upper bound: 12.5400337
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5632719, upper bound: 12.5117545
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5341931, upper bound: 12.5292782
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5694125, upper bound: 12.4940327
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5284671, upper bound: 12.4850242
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5404283, upper bound: 12.4732533
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5490523, upper bound: 12.5296854
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5603954, upper bound: 12.5183289
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5673733, upper bound: 12.5061928
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5508465, upper bound: 12.5227227
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5479836, upper bound: 12.5560190
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5433464, upper bound: 12.5606515
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5599513, upper bound: 12.5449543
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5605347, upper bound: 12.5443683
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5441861, upper bound: 12.5004343
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5612490, upper bound: 12.4833679
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5406509, upper bound: 12.5670388
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5415800, upper bound: 12.5661128
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5447459, upper bound: 12.5632768
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5453838, upper bound: 12.5626388
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5367829, upper bound: 12.5676464
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5374208, upper bound: 12.5670108
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5120873, upper bound: 12.5625793
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5189382, upper bound: 12.5557270
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5102173, upper bound: 12.5673512
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5140085, upper bound: 12.5635610
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5419198, upper bound: 12.5246487
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5611783, upper bound: 12.5053780
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5598685, upper bound: 12.5251621
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5605904, upper bound: 12.5244396
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5378856, upper bound: 12.5492224
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5264479, upper bound: 12.5606579
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5212867, upper bound: 12.5597005
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5119926, upper bound: 12.5712826
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5222689, upper bound: 12.5700741
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.88
Output dim: 14, lower bound: -12.5210145, upper bound: 12.5713117
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.88
Output dim: 14, lower bound: -12.5476754, upper bound: 12.5643035
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.88
Output dim: 14, lower bound: -12.5478023, upper bound: 12.5641768
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.88
Output dim: 14, lower bound: -12.5453301, upper bound: 12.5631840
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.88
Output dim: 14, lower bound: -12.5365538, upper bound: 12.5668200

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 23.45 + 1788.19 = 1811.64 seconds
