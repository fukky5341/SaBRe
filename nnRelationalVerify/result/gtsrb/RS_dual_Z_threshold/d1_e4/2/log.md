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
execution time: IAR + RelationalAnalysis = 2.77 + 20.46 = 23.23 seconds
status: Status.UNKNOWN
relational distance
Output dim: 14, lower bound: -12.5728910, upper bound: 12.5728909

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5632410, upper bound: 12.5658605
time: 9.22 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5658605, upper bound: 12.5632410
time: 24.01 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 33.35 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 33.35
Output dim: 14, lower bound: -12.5632410, upper bound: 12.5658605
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 33.35
Output dim: 14, lower bound: -12.5658605, upper bound: 12.5632410

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8994942, 13.9010925
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5085602, 8.5089664
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4611969, 13.4619293
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0315781, 12.0325279
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6902466, 14.6886292
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2156944, 15.2148972
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6388702, 13.6396561
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1122208, 12.1138744
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0370560, 13.0379562
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9527130, 20.9549255
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0365791, 15.0362129
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6313019, 16.6292191
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7742538, 26.7760086
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8322906, 14.8345299
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2538719, 17.2563286
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4319992, 14.4316139
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5555878, 14.5547180
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6372757, 12.6360855
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9731903, 14.9710579
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3401489, 14.3414001
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2775345, 9.2766685
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8234940, 13.8234863
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3729515, 19.3719788
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2317505, 13.2307587
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9780731, 14.9775085
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5620689, 13.5621185
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6627579, 14.6621780
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3210945, 13.3198738
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6748848, 16.6708527
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1713486, 14.1682014
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0246849, 13.0204277
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1268539, 15.1220894
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9269791, 18.9241409
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3949242, 18.3889694
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.1002502, 16.0946503
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3520126, 14.3520107
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7043762, 16.7029076
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2715340, 17.2730713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1729

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5483765, upper bound: 12.5652900
time: 10.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5627228, upper bound: 12.5513893
time: 8.23 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9000626, 13.8994942
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5087318, 8.5085621
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4614334, 13.4611931
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0319748, 12.0315781
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6886292, 14.6887856
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2149010, 15.2154999
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6391602, 13.6388702
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1128159, 12.1122227
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0374336, 13.0370560
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9535446, 20.9527130
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0362129, 15.0364494
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6292191, 16.6293945
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7749405, 26.7742462
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8331146, 14.8322945
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2550125, 17.2538719
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4316139, 14.4317741
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5547180, 14.5550919
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6360855, 12.6365433
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9710541, 14.9717941
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3402557, 14.3401489
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2766685, 9.2769775
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8234863, 13.8234940
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3719749, 19.3724365
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2307587, 13.2311211
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9775085, 14.9777222
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5620842, 13.5620689
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6621780, 14.6624260
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3198738, 13.3205414
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6708488, 16.6724014
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1681976, 14.1694450
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0204277, 13.0219536
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1220932, 15.1241570
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9241409, 18.9252701
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3889732, 18.3915100
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0946503, 16.0968819
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3520126, 14.3520813
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7029037, 16.7036476
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2716637, 17.2715302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1729

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5513893, upper bound: 12.5627227
time: 11.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5652901, upper bound: 12.5483765
time: 9.19 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.44 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.44
Output dim: 14, lower bound: -12.5483765, upper bound: 12.5652900
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.44
Output dim: 14, lower bound: -12.5627228, upper bound: 12.5513893
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.44
Output dim: 14, lower bound: -12.5513893, upper bound: 12.5627227
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.44
Output dim: 14, lower bound: -12.5652901, upper bound: 12.5483765

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8925629, 13.8949203
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5089874, 8.5093498
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4589157, 13.4599380
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0285721, 12.0296688
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6930618, 14.6902771
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2034225, 15.2012825
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6390152, 13.6397858
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1052742, 12.1075974
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0284805, 13.0302544
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9406891, 20.9446411
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0346375, 15.0341301
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6335907, 16.6310844
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7636261, 26.7666931
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8209305, 14.8240013
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2383919, 17.2426567
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4265652, 14.4254646
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5507202, 14.5491295
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6297073, 12.6275330
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9680061, 14.9649391
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3411865, 14.3432312
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2732925, 9.2721977
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8236847, 13.8236847
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3707657, 19.3697815
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2319908, 13.2300835
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9779205, 14.9773750
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5623436, 13.5624313
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6571236, 14.6557541
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3128777, 13.3105278
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6586113, 16.6523285
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1592789, 14.1544495
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0061760, 12.9994049
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1051254, 15.0975380
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9183502, 18.9145546
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3620377, 18.3517303
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0768509, 16.0680046
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3518295, 14.3518772
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6961670, 16.6935616
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2774124, 17.2799377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 689

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5475464, upper bound: 12.5484175
time: 13.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5357121, upper bound: 12.5649888
time: 8.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8933525, 13.8941612
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5089493, 8.5093880
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4592056, 13.4596558
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0287323, 12.0295200
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6919022, 14.6914902
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2020874, 15.2025452
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6389999, 13.6397972
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1059837, 12.1069298
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0293541, 13.0293808
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9425049, 20.9429016
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0344963, 15.0342712
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6331711, 16.6315079
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7649384, 26.7653809
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8217621, 14.8231621
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2402763, 17.2408485
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4258480, 14.4261799
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5500031, 14.5498428
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6287231, 12.6285362
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9670677, 14.9659119
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3419800, 14.3424339
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2730637, 9.2724342
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8236923, 13.8236771
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3707657, 19.3697891
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2310753, 13.2309990
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9779358, 14.9773636
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5623817, 13.5623932
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6563377, 14.6565475
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3117485, 13.3116570
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6563606, 16.6545753
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1576004, 14.1561241
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0036659, 13.0019684
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1023026, 15.1004829
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9173965, 18.9155159
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3576813, 18.3562775
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0736008, 16.0712509
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3518829, 14.3518238
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6950302, 16.6947021
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2783966, 17.2789497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 689

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5624216, upper bound: 12.5387003
time: 11.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5458459, upper bound: 12.5505590
time: 17.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8931351, 13.8933525
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5091553, 8.5089493
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4591675, 13.4592056
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0289612, 12.0287342
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6914902, 14.6904335
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2025452, 15.2018890
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6393051, 13.6389999
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1058693, 12.1059799
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0288544, 13.0293541
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9415359, 20.9425049
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0342712, 15.0343704
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6315079, 16.6312599
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7643280, 26.7649384
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8217545, 14.8217659
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2395325, 17.2402725
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4261799, 14.4256248
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5498352, 14.5494919
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6285362, 12.6279945
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9659081, 14.9656792
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3413010, 14.3419838
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2724304, 9.2725182
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8236771, 13.8236847
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3697891, 19.3702393
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2309990, 13.2304497
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9773636, 14.9775887
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5623627, 13.5623817
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6565437, 14.6560097
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3116570, 13.3111916
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6545753, 16.6538773
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1561203, 14.1556854
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0019646, 13.0009384
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1004868, 15.0996094
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9155121, 18.9156837
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3562698, 18.3542633
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0712509, 16.0702324
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3518219, 14.3519459
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6947021, 16.6943016
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2775421, 17.2783966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 689

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5505591, upper bound: 12.5458458
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5387003, upper bound: 12.5624215
time: 7.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8939247, 13.8925629
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5091171, 8.5089874
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4594498, 13.4589195
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0291290, 12.0285702
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6902847, 14.6916466
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2012863, 15.2031479
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6392975, 13.6390114
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1065712, 12.1052780
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0297279, 13.0284805
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9433441, 20.9406891
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0341301, 15.0345116
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6310806, 16.6316833
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7656403, 26.7636185
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8225937, 14.8209305
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2414169, 17.2383881
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4254627, 14.4263420
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5491333, 14.5502052
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6275330, 12.6289978
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9649391, 14.9666481
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3420944, 14.3411865
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2721977, 9.2727547
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8236847, 13.8236809
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3697891, 19.3702469
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2300835, 13.2313652
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9773788, 14.9775734
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5624008, 13.5623436
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6557579, 14.6567955
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3105278, 13.3123207
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6523247, 16.6561279
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1544495, 14.1573639
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9994087, 13.0035019
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0975418, 15.1025543
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9145584, 18.9166412
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3517303, 18.3588104
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0680008, 16.0734787
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3518753, 14.3518925
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6935654, 16.6954422
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2785339, 17.2774086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 689

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5649889, upper bound: 12.5357120
time: 8.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5484176, upper bound: 12.5475463
time: 6.00 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.66 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 16.66
Output dim: 14, lower bound: -12.5475464, upper bound: 12.5484175
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.66
Output dim: 14, lower bound: -12.5357121, upper bound: 12.5649888
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.66
Output dim: 14, lower bound: -12.5624216, upper bound: 12.5387003
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 16.66
Output dim: 14, lower bound: -12.5458459, upper bound: 12.5505590
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 16.66
Output dim: 14, lower bound: -12.5505591, upper bound: 12.5458458
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.66
Output dim: 14, lower bound: -12.5387003, upper bound: 12.5624215
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.66
Output dim: 14, lower bound: -12.5649889, upper bound: 12.5357120
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 16.66
Output dim: 14, lower bound: -12.5484176, upper bound: 12.5475463

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8888893, 13.8915100
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5013542, 8.4990959
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4595871, 13.4608345
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0373993, 12.0354843
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6830215, 14.6787682
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1336136, 15.1217117
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6207199, 13.6175079
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1067734, 12.1091175
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0002556, 13.0055008
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9484558, 20.9548111
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0254135, 15.0236206
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6278038, 16.6275558
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7332916, 26.7400970
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8232841, 14.8256950
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2372131, 17.2416229
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4229622, 14.4214973
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5476379, 14.5457840
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6238022, 12.6202049
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9549942, 14.9528809
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3149757, 14.3202820
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2751122, 9.2756538
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8134003, 13.8151779
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3249054, 19.3319397
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2370300, 13.2343521
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9548111, 14.9571266
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5629959, 13.5638885
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6568947, 14.6535301
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2845421, 13.2768784
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6313248, 16.6212234
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1220932, 14.1109123
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9946213, 12.9853401
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1012497, 15.0931053
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9247742, 18.9233704
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3280106, 18.3129272
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0784607, 16.0693283
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3456230, 14.3452225
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6843109, 16.6790466
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2745743, 17.2764549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1712

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5226077, upper bound: 12.5649890
time: 8.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5356293, upper bound: 12.5505438
time: 8.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8899460, 13.8904877
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4986954, 8.5017548
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4600983, 13.4603195
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0345459, 12.0383453
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6803894, 14.6814537
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1225166, 15.1327400
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6167221, 13.6215096
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1075058, 12.1084232
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0046005, 13.0011559
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9526749, 20.9506683
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0239906, 15.0250473
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6296425, 16.6257172
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7383423, 26.7350464
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8234596, 14.8255196
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2392426, 17.2396698
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4218788, 14.4225769
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5466614, 14.5467567
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6213951, 12.6226349
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9550095, 14.9528999
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3190346, 14.3162270
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2765198, 9.2742538
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8151855, 13.8133926
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3329163, 19.3239288
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2353439, 13.2360382
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9576950, 14.9542580
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5638390, 13.5630493
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6541100, 14.6563110
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2780991, 13.2833252
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6252594, 16.6272888
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1140671, 14.1189384
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9896011, 12.9904099
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0978699, 15.0966034
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9262085, 18.9219322
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3188705, 18.3222504
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0749283, 16.0728607
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3452263, 14.3456192
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6805191, 16.6828461
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2749176, 17.2761192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1712

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5479763, upper bound: 12.5386175
time: 8.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5624217, upper bound: 12.5255980
time: 13.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8894615, 13.8899460
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5015182, 8.4986954
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4598312, 13.4600983
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0377884, 12.0345497
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6814499, 14.6789169
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1327400, 15.1223183
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6210098, 13.6167221
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1073608, 12.1075039
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0006294, 13.0046005
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9493027, 20.9526749
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0250473, 15.0238647
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6257133, 16.6277351
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7339935, 26.7383423
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8241081, 14.8234596
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2383499, 17.2392426
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4225731, 14.4216576
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5467529, 14.5461502
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6226349, 12.6206703
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9529037, 14.9536133
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3150826, 14.3190346
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2742538, 9.2759743
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8133926, 13.8151855
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3239288, 19.3324051
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2360382, 13.2347260
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9542618, 14.9573479
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5630150, 13.5638390
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6563148, 14.6537704
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2833214, 13.2775459
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6272888, 16.6227684
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1189423, 14.1121521
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9904099, 12.9868698
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0966034, 15.0951691
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9219360, 18.9244919
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3222427, 18.3154678
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0728607, 16.0715675
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3456192, 14.3452892
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6828461, 16.6797905
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2747116, 17.2749138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1712

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5255980, upper bound: 12.5624216
time: 10.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5386175, upper bound: 12.5479763
time: 7.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8905182, 13.8888893
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4988632, 8.5013542
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4603500, 13.4595833
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0349426, 12.0373955
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6787720, 14.6815987
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1217117, 15.1333427
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6170120, 13.6207237
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1080933, 12.1067715
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0049706, 13.0002556
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9535217, 20.9484558
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0236206, 15.0252838
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6275597, 16.6258926
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7390442, 26.7332840
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8242836, 14.8232841
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2403793, 17.2372093
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4214973, 14.4227352
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5457916, 14.5471230
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6202049, 12.6231003
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9528732, 14.9536362
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3191338, 14.3149757
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2756538, 9.2745705
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8151779, 13.8134003
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3319397, 19.3243942
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2343521, 13.2364120
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9571304, 14.9544754
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5638542, 13.5629959
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6535301, 14.6565514
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2768784, 13.2839928
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6212234, 16.6288376
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1109161, 14.1201782
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9853439, 12.9919357
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0931015, 15.0986710
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9233704, 18.9230576
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3129196, 18.3247910
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0693283, 16.0750999
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3452225, 14.3456841
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6790466, 16.6835899
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2750473, 17.2745781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1712

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5505439, upper bound: 12.5356292
time: 7.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5649890, upper bound: 12.5226076
time: 6.55 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 16.29 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 14, lower bound: -12.5226077, upper bound: 12.5649890
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 16.29
Output dim: 14, lower bound: -12.5356293, upper bound: 12.5505438
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 16.29
Output dim: 14, lower bound: -12.5479763, upper bound: 12.5386175
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 14, lower bound: -12.5624217, upper bound: 12.5255980
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 14, lower bound: -12.5255980, upper bound: 12.5624216
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 16.29
Output dim: 14, lower bound: -12.5386175, upper bound: 12.5479763
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 16.29
Output dim: 14, lower bound: -12.5505439, upper bound: 12.5356292
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 14, lower bound: -12.5649890, upper bound: 12.5226076

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8870583, 13.8898811
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5029793, 8.4997044
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4622498, 13.4639053
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0373535, 12.0353661
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6708374, 14.6646919
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1077919, 15.0922012
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6152573, 13.6107941
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0942230, 12.0984497
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9944305, 13.0004005
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9627151, 20.9718399
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0139046, 15.0104294
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6083031, 16.6058998
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7201004, 26.7285538
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8220329, 14.8247185
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2274437, 17.2330780
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4212189, 14.4195843
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5515976, 14.5495224
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6275101, 12.6231155
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9579773, 14.9555702
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2973099, 14.3048553
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2740173, 9.2750931
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8048172, 13.8078041
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3128204, 19.3222733
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2363548, 13.2336426
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9456558, 14.9491119
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5623360, 13.5632820
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6620750, 14.6579437
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2665176, 13.2560158
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6008377, 16.5863762
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1073647, 14.0936012
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9739304, 12.9606400
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0872726, 15.0756531
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9242439, 18.9223213
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2853241, 18.2641296
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0408707, 16.0263710
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3464165, 14.3460484
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6725922, 16.6655960
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2709312, 17.2741852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5223324, upper bound: 12.5546356
time: 10.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5122368, upper bound: 12.5647144
time: 7.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8883171, 13.8886566
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4993057, 8.5033817
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4631729, 13.4629784
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0344315, 12.0383034
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6663132, 14.6692696
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0930061, 15.1069145
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6100082, 13.6160469
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0968323, 12.0958786
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9995003, 12.9953308
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9697037, 20.9649277
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0107956, 15.0135345
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6079826, 16.6062202
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7267990, 26.7218552
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8224831, 14.8242607
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2306938, 17.2299042
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4199677, 14.4208393
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5504074, 14.5507126
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6243095, 12.6263428
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9577026, 14.9558830
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3036041, 14.2985611
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2759590, 9.2731590
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8078079, 13.8048134
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3232498, 19.3118439
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2346344, 13.2353630
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9496765, 14.9450989
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5632324, 13.5623856
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6585274, 14.6614914
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2572327, 13.2653008
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5904160, 16.5968018
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0967522, 14.1042099
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9648972, 12.9697227
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0804214, 15.0826263
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9251595, 18.9214134
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2700729, 18.2795563
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0319672, 16.0352745
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3460541, 14.3464127
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6670685, 16.6711273
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2726402, 17.2724724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5621472, upper bound: 12.5152282
time: 8.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5520679, upper bound: 12.5253227
time: 9.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8876266, 13.8883171
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5031471, 8.4993038
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4624939, 13.4631729
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0377426, 12.0344276
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6692734, 14.6648483
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1069145, 15.0927963
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6155624, 13.6100082
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0948105, 12.0968323
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9948082, 12.9995003
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9635544, 20.9697037
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0135345, 15.0106735
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6062202, 16.6060715
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7208023, 26.7267990
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8228569, 14.8224831
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2285957, 17.2306938
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4208374, 14.4197426
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5507126, 14.5498886
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6263428, 12.6235771
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9558792, 14.9563103
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2974167, 14.3036041
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2731590, 9.2754097
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8048172, 13.8078156
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3118439, 19.3227310
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2353630, 13.2340126
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9450989, 14.9493294
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5623550, 13.5632324
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6614876, 14.6581879
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2652969, 13.2566833
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5968018, 16.5879288
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1042061, 14.0948410
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9697189, 12.9621658
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0826263, 15.0777245
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9214058, 18.9234467
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2795563, 18.2666550
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0352783, 16.0286140
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3464127, 14.3461132
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6711273, 16.6663399
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2710533, 17.2726440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5253227, upper bound: 12.5520678
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5152283, upper bound: 12.5621471
time: 6.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8888855, 13.8870583
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4994698, 8.5029774
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4634247, 13.4622459
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0348206, 12.0373535
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6646957, 14.6694260
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0922012, 15.1075134
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6102982, 13.6152611
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0974197, 12.0942230
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9998779, 12.9944305
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9705429, 20.9627151
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0104294, 15.0137787
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6058998, 16.6063919
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7275009, 26.7200928
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8233070, 14.8220291
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2318459, 17.2274437
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4195862, 14.4209976
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5495224, 14.5510788
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6231155, 12.6268005
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9555740, 14.9566231
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3037109, 14.2973137
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2750931, 9.2734718
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8078079, 13.8048210
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3222733, 19.3123016
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2336426, 13.2357330
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9491119, 14.9453125
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5632515, 13.5623360
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6579475, 14.6617355
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2560120, 13.2659683
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5863800, 16.5983505
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0936012, 14.1054497
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9606400, 12.9712486
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0756531, 15.0846977
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9223213, 18.9225388
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2641296, 18.2820892
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0263748, 16.0375137
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3460464, 14.3464775
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6655960, 16.6718674
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2727623, 17.2709312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5647145, upper bound: 12.5122367
time: 8.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5546356, upper bound: 12.5223323
time: 7.80 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 18.93 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.93
Output dim: 14, lower bound: -12.5223324, upper bound: 12.5546356
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 14, lower bound: -12.5122368, upper bound: 12.5647144
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 14, lower bound: -12.5621472, upper bound: 12.5152282
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.93
Output dim: 14, lower bound: -12.5520679, upper bound: 12.5253227
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.93
Output dim: 14, lower bound: -12.5253227, upper bound: 12.5520678
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 14, lower bound: -12.5152283, upper bound: 12.5621471
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.93
Output dim: 14, lower bound: -12.5647145, upper bound: 12.5122367
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.93
Output dim: 14, lower bound: -12.5546356, upper bound: 12.5223323

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8653793, 13.8708687
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4971046, 8.4943962
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4591675, 13.4611740
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0323334, 12.0307961
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6811295, 14.6735458
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1133080, 15.0976639
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6088257, 13.6048775
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0930252, 12.0972729
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9618301, 12.9718056
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9518356, 20.9622498
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0277863, 15.0266228
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5849609, 16.5853043
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7025833, 26.7131500
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7903519, 14.7969322
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2190018, 17.2268944
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4218254, 14.4178543
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5387115, 14.5348358
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6170082, 12.6111450
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9401321, 14.9352341
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2905464, 14.2972374
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2493591, 9.2469826
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7909241, 13.7919617
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2961769, 19.3035126
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2160454, 13.2102242
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9336166, 14.9355240
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5542450, 13.5543137
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6441803, 14.6375580
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2732620, 13.2625580
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6005859, 16.5861053
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0867958, 14.0701561
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9576111, 12.9421501
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0679550, 15.0540466
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9259109, 18.9239578
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2523651, 18.2265549
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0336113, 16.0187492
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3585281, 14.3593941
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6717987, 16.6647415
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2865715, 17.2919884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4698247, upper bound: 12.5645319
time: 15.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5120547, upper bound: 12.5223091
time: 7.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8693008, 13.8669777
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4939919, 8.4975052
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4604416, 13.4598999
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0298615, 12.0332832
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6751633, 14.6795578
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0984650, 15.1124306
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6040955, 13.6096115
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0956573, 12.0946789
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9709091, 12.9627304
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9601135, 20.9540482
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0269890, 15.0274200
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5873871, 16.5828819
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7114029, 26.7043381
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7947006, 14.7925797
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2245102, 17.2214622
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4182396, 14.4214439
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5357132, 14.5378342
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6123352, 12.6158409
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9373627, 14.9380302
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2959862, 14.2917976
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2478485, 9.2485008
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7919693, 13.7909164
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3044930, 19.2951965
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2112160, 13.2150536
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9360809, 14.9330559
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5542641, 13.5542946
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6381378, 14.6436005
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2637787, 13.2720413
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5901489, 16.5965462
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0733070, 14.0836449
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9464111, 12.9534035
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0588150, 15.0633087
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9267960, 18.9230728
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2325134, 18.2466049
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0243492, 16.0280151
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3593979, 14.3585243
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6662140, 16.6703339
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2904472, 17.2881126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5197422, upper bound: 12.5150461
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5619646, upper bound: 12.4728162
time: 8.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8659515, 13.8693008
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4972725, 8.4939919
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4594116, 13.4604378
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0327301, 12.0298615
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6795578, 14.6736946
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1124306, 15.0982628
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6091156, 13.6040916
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0936203, 12.0956593
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9622040, 12.9709053
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9526596, 20.9601135
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0274200, 15.0268593
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5828857, 16.5854797
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7032547, 26.7113953
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7911682, 14.7946968
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2201462, 17.2245102
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4214439, 14.4180145
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5378342, 14.5352020
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6158409, 12.6116066
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9380341, 14.9359741
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2906532, 14.2959900
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2485008, 9.2473068
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7909164, 13.7919731
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2951927, 19.3039780
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2150536, 13.2105904
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9330597, 14.9357376
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5542641, 13.5542641
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6436005, 14.6378098
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2720413, 13.2632256
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5965424, 16.5876579
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0836449, 14.0713959
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9533997, 12.9436798
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0633087, 15.0561180
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9230728, 18.9250832
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2465973, 18.2290878
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0280190, 16.0209846
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3585243, 14.3594570
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6703339, 16.6654892
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2867088, 17.2904472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4728162, upper bound: 12.5619645
time: 13.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5150461, upper bound: 12.5197422
time: 10.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8698730, 13.8653793
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4941597, 8.4971046
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4606934, 13.4591637
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0302582, 12.0323334
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6735458, 14.6797028
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0976639, 15.1130333
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6043854, 13.6088257
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0962524, 12.0930233
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9712791, 12.9618301
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9609451, 20.9518356
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0266228, 15.0276566
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5853043, 16.5830574
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7120743, 26.7025757
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7955170, 14.7903481
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2256546, 17.2190018
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4178505, 14.4216042
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5348358, 14.5381966
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6111450, 12.6163025
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9352341, 14.9387741
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2960930, 14.2905502
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2469826, 9.2488251
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7919617, 13.7909279
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3035164, 19.2956619
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2102242, 13.2154160
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9355240, 14.9332695
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5542831, 13.5542450
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6375580, 14.6438484
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2625580, 13.2727089
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5861053, 16.5980949
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0701561, 14.0848885
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9421463, 12.9549332
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0540466, 15.0653801
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9239578, 18.9241982
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2265625, 18.2491379
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0187492, 16.0302544
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3593941, 14.3585892
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6647491, 16.6710815
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2905846, 17.2865753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5223092, upper bound: 12.5120546
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5645319, upper bound: 12.4698246
time: 6.76 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 15.25 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.25
Output dim: 14, lower bound: -12.4698247, upper bound: 12.5645319
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.25
Output dim: 14, lower bound: -12.5120547, upper bound: 12.5223091
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.25
Output dim: 14, lower bound: -12.5197422, upper bound: 12.5150461
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.25
Output dim: 14, lower bound: -12.5619646, upper bound: 12.4728162
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.25
Output dim: 14, lower bound: -12.4728162, upper bound: 12.5619645
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.25
Output dim: 14, lower bound: -12.5150461, upper bound: 12.5197422
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.25
Output dim: 14, lower bound: -12.5223092, upper bound: 12.5120546
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.25
Output dim: 14, lower bound: -12.5645319, upper bound: 12.4698246

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8582153, 13.8628426
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4909096, 8.4869099
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4583092, 13.4601479
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0314484, 12.0310707
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6795273, 14.6717262
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1242981, 15.1097908
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6139297, 13.6093712
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1060181, 12.1078720
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9682732, 12.9789200
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9486542, 20.9592438
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0243149, 15.0243568
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6187057, 16.6236343
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.6996307, 26.7132492
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7878265, 14.7919464
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2180824, 17.2262535
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4245834, 14.4194489
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5344162, 14.5299835
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6030312, 12.5979805
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9406052, 14.9378128
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2925644, 14.2989960
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2467842, 9.2441139
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7875862, 13.7882500
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2536430, 19.2583771
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2195816, 13.2133598
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9146614, 14.9191360
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5553589, 13.5555534
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6493225, 14.6420174
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2884293, 13.2798576
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5646820, 16.5462646
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0878792, 14.0654030
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9587021, 12.9421005
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0493927, 15.0374413
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9274025, 18.9234734
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2417717, 18.2149200
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0275078, 16.0114822
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3940964, 14.3891792
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6937790, 16.6838188
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3157501, 17.3189888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4592495, upper bound: 12.5641726
time: 14.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4694658, upper bound: 12.5539354
time: 22.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8612785, 13.8598175
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4865074, 8.4913120
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4594078, 13.4590378
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0301361, 12.0323944
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6733475, 14.6779518
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1105919, 15.1234245
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6085815, 13.6147156
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1062622, 12.1076698
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9780235, 12.9691696
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9571075, 20.9508667
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0247231, 15.0239487
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6257172, 16.6166229
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7114868, 26.7013931
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7897110, 14.7900581
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2238731, 17.2205467
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4198303, 14.4241982
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5308609, 14.5335464
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5991707, 12.6018639
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9399490, 14.9385147
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2977448, 14.2938118
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2449760, 9.2459259
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7882576, 13.7875824
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2593575, 19.2526627
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2143478, 13.2185898
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9196968, 14.9141006
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5555038, 13.5554085
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6425934, 14.6487427
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2810783, 13.2872086
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5502930, 16.5606461
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0685539, 14.0847244
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9463577, 12.9544907
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0422058, 15.0447464
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9263115, 18.9245605
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2208672, 18.2360077
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0170784, 16.0219116
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3891830, 14.3940926
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6852951, 16.6923027
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3174438, 17.3172913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5513673, upper bound: 12.4724574
time: 6.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5616054, upper bound: 12.4622411
time: 7.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8587875, 13.8612785
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4910736, 8.4865055
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4585533, 13.4594116
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0318451, 12.0301361
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6779556, 14.6718826
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1234245, 15.1103973
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6142273, 13.6085854
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1066132, 12.1062584
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9686508, 12.9780197
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9495010, 20.9571075
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0239487, 15.0245934
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6166229, 16.6238174
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7003326, 26.7114944
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7886505, 14.7897110
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2192345, 17.2238731
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4241943, 14.4196091
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5335464, 14.5303535
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6018639, 12.5984344
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9385147, 14.9385529
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2926559, 14.2977486
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2459259, 9.2444305
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7875786, 13.7882576
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2526588, 19.2588348
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2185898, 13.2137260
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9140968, 14.9193459
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5553780, 13.5555038
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6487427, 14.6422653
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2872086, 13.2805214
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5606384, 16.5478058
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0847282, 14.0666428
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9544907, 12.9436302
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0447464, 15.0395126
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9245567, 18.9245987
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2360039, 18.2174530
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0219078, 16.0137138
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3940926, 14.3892479
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6922989, 16.6845627
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3158798, 17.3174477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4622412, upper bound: 12.5616053
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4724574, upper bound: 12.5513672
time: 13.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8618507, 13.8582153
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4866714, 8.4909077
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4596672, 13.4583054
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0305328, 12.0314445
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6717300, 14.6781082
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1097908, 15.1240273
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6088791, 13.6139297
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1068573, 12.1060181
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9784012, 12.9682732
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9579620, 20.9486542
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0243568, 15.0241890
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6236343, 16.6168060
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7121887, 26.6996307
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7905426, 14.7878227
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2250175, 17.2180862
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4194489, 14.4243584
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5299911, 14.5339127
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5979805, 12.6023178
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9378128, 14.9392509
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2978439, 14.2925606
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2441139, 9.2462425
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7882500, 13.7875900
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2583733, 19.2531204
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2133560, 13.2189560
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9191322, 14.9143105
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5555229, 13.5553589
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6420135, 14.6489906
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2798576, 13.2878723
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5462646, 16.5621910
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0654030, 14.0859680
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9421005, 12.9560204
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0374374, 15.0468216
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9234734, 18.9256859
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2149162, 18.2385406
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0114784, 16.0241432
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3891792, 14.3941612
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6838150, 16.6930466
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3175735, 17.3157501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5539354, upper bound: 12.4694658
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5641727, upper bound: 12.4592495
time: 7.05 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 16.43 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.43
Output dim: 14, lower bound: -12.4592495, upper bound: 12.5641726
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 16.43
Output dim: 14, lower bound: -12.4694658, upper bound: 12.5539354
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.43
Output dim: 14, lower bound: -12.5513673, upper bound: 12.4724574
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.43
Output dim: 14, lower bound: -12.5616054, upper bound: 12.4622411
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.43
Output dim: 14, lower bound: -12.4622412, upper bound: 12.5616053
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 16.43
Output dim: 14, lower bound: -12.4724574, upper bound: 12.5513672
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.43
Output dim: 14, lower bound: -12.5539354, upper bound: 12.4694658
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.43
Output dim: 14, lower bound: -12.5641727, upper bound: 12.4592495

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8541985, 13.8593292
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4897270, 8.4858856
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4607468, 13.4625168
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0202293, 12.0211296
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6907196, 14.6816750
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1331863, 15.1191711
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6108932, 13.6066628
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1068916, 12.1087494
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9571114, 12.9692268
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9380341, 20.9498062
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0146713, 15.0164719
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6101646, 16.6159134
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.6874619, 26.7025681
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7779160, 14.7832565
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2004395, 17.2107162
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4223328, 14.4152451
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5232506, 14.5172539
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6007767, 12.5954056
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9354210, 14.9319077
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2848015, 14.2901154
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2296829, 9.2245483
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7841225, 13.7843933
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2374420, 19.2398987
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2037468, 13.1954803
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9120331, 14.9161491
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5566521, 13.5569038
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6416664, 14.6331749
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2884598, 13.2804298
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5596504, 16.5408020
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0844040, 14.0614357
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9500771, 12.9324112
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0373383, 15.0236740
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9087791, 18.9023552
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2241554, 18.1948853
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0111351, 15.9932594
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3963318, 14.3912468
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6859055, 16.6749458
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3280792, 17.3327866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 690

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4582417, upper bound: 12.5603883
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4565469, upper bound: 12.5626350
time: 8.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8577652, 13.8558006
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4854813, 8.4901276
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4617844, 13.4614830
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0201912, 12.0211830
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6832962, 14.6891479
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1199722, 15.1323166
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6058731, 13.6116791
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1071281, 12.1085434
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9683266, 12.9580116
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9476700, 20.9402466
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0168381, 15.0143051
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6179924, 16.6080856
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7008286, 26.6892319
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7810287, 14.7801476
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2083359, 17.2028961
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4156303, 14.4219475
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5181313, 14.5223732
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5965958, 12.5996094
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9340401, 14.9333267
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2888680, 14.2860527
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2254181, 9.2288246
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7844048, 13.7841187
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2408829, 19.2364655
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.1964722, 13.2027550
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9167175, 14.9114761
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5568542, 13.5567017
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6337547, 14.6410828
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2816505, 13.2872391
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5448418, 16.5556107
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0645905, 14.0812531
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9366722, 12.9458656
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0284424, 15.0326920
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9051933, 18.9059448
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2008400, 18.2183990
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -15.9988594, 16.0055313
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3912506, 14.3963299
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6764145, 16.6844406
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3312454, 17.3296280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 690

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5600730, upper bound: 12.4595466
time: 15.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5578237, upper bound: 12.4612298
time: 26.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8547707, 13.8577614
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4898911, 8.4854851
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4609985, 13.4617844
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0206299, 12.0201912
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6891479, 14.6818237
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1323166, 15.1197624
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6111832, 13.6058769
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1074867, 12.1071320
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9574890, 12.9683266
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9388733, 20.9476700
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0143051, 15.0167084
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6080894, 16.6160927
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.6881638, 26.7008209
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7787399, 14.7810249
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2015839, 17.2083321
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4219475, 14.4154053
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5223732, 14.5176201
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5996094, 12.5958633
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9333229, 14.9326477
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2849083, 14.2888680
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2288246, 9.2248650
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7841148, 13.7844009
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2364655, 19.2403564
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2027550, 13.1958504
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9114838, 14.9163666
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5566673, 13.5568542
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6410866, 14.6334267
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2872391, 13.2810974
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5556145, 16.5423470
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0812531, 14.0626831
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9458656, 12.9339447
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0326920, 15.0257416
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9059410, 18.9034805
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2184029, 18.1974182
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0055351, 15.9954948
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3963318, 14.3913078
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6844406, 16.6756897
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3282089, 17.3312454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 690

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4612298, upper bound: 12.5578237
time: 18.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4595467, upper bound: 12.5600729
time: 8.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8583336, 13.8541985
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4856453, 8.4897270
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4620285, 13.4607468
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0205917, 12.0202293
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6816711, 14.6892967
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1191711, 15.1329079
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6061707, 13.6108932
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1077309, 12.1068916
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9687004, 12.9571114
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9485092, 20.9380341
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0164719, 15.0145454
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6159172, 16.6082611
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7015152, 26.6874695
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7818527, 14.7779160
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2094727, 17.2004395
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4152451, 14.4221077
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5172539, 14.5227394
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5954056, 12.6000671
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9319038, 14.9340668
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2889748, 14.2848053
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2245483, 9.2291412
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7843971, 13.7841225
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2398987, 19.2369232
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.1954803, 13.2031212
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9161530, 14.9116936
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5568695, 13.5566521
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6331749, 14.6413345
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2804298, 13.2879066
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5408058, 16.5571518
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0614395, 14.0825005
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9324150, 12.9473991
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0236740, 15.0347633
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9023552, 18.9070702
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.1948891, 18.2209320
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -15.9932594, 16.0077667
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3912468, 14.3963909
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6749420, 16.6851845
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3313751, 17.3280869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 690

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5626351, upper bound: 12.4565469
time: 10.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5603884, upper bound: 12.4582416
time: 10.68 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 23.85 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 23.85
Output dim: 14, lower bound: -12.4582417, upper bound: 12.5603883
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 23.85
Output dim: 14, lower bound: -12.4565469, upper bound: 12.5626350
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 23.85
Output dim: 14, lower bound: -12.5600730, upper bound: 12.4595466
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 23.85
Output dim: 14, lower bound: -12.5578237, upper bound: 12.4612298
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 23.85
Output dim: 14, lower bound: -12.4612298, upper bound: 12.5578237
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 23.85
Output dim: 14, lower bound: -12.4595467, upper bound: 12.5600729
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 23.85
Output dim: 14, lower bound: -12.5626351, upper bound: 12.4565469
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 23.85
Output dim: 14, lower bound: -12.5603884, upper bound: 12.4582416

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8541565, 13.8592339
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4889755, 8.4867039
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4606705, 13.4624443
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0201302, 12.0228081
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6906586, 14.6816978
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1321030, 15.1200600
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6106033, 13.6075706
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1066093, 12.1098785
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9577255, 12.9687920
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9378738, 20.9496460
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0149460, 15.0164337
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6119156, 16.6140327
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.6873932, 26.7023239
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7771378, 14.7837982
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2003937, 17.2102318
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4221802, 14.4153385
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5231628, 14.5172272
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6006927, 12.5953865
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9355850, 14.9314117
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2847443, 14.2899513
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2297325, 9.2245026
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7837410, 13.7844200
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2376060, 19.2383270
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2037430, 13.1954880
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9122963, 14.9155273
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5561371, 13.5565453
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6408920, 14.6342087
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2884445, 13.2804260
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5596504, 16.5408058
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0833664, 14.0622559
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9500427, 12.9324455
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0380096, 15.0224876
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9117661, 18.9021606
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2239494, 18.1948624
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0121689, 15.9929085
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3964806, 14.3910160
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6856918, 16.6748123
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3276215, 17.3326340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 955

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4388525, upper bound: 12.5602652
time: 7.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4581235, upper bound: 12.5409971
time: 8.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8541985, 13.8592834
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4897270, 8.4851322
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4606705, 13.4625168
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0202293, 12.0210228
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6907196, 14.6816177
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1331863, 15.1180840
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6108932, 13.6063728
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1068916, 12.1084671
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9566765, 12.9692268
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9380341, 20.9496384
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0146370, 15.0164719
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6082840, 16.6159134
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.6872253, 26.7025681
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7779160, 14.7824821
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2004395, 17.2106705
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4223328, 14.4150963
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5232239, 14.5172539
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6007576, 12.5954056
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9349289, 14.9319077
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2846375, 14.2901154
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2296371, 9.2245483
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7841225, 13.7840080
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2358589, 19.2398987
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2037468, 13.1954803
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9114113, 14.9161491
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5566521, 13.5563889
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6416664, 14.6324081
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2884598, 13.2804146
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5596504, 16.5408020
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0844040, 14.0603981
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9500771, 12.9323769
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0361481, 15.0236740
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9085922, 18.9023552
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2241325, 18.1948853
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0107803, 15.9932594
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3961029, 14.3912468
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6857834, 16.6749458
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3280792, 17.3323212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 955

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4371578, upper bound: 12.5625121
time: 12.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4564288, upper bound: 12.5432529
time: 9.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8582916, 13.8541336
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4848938, 8.4905453
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4619980, 13.4606743
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0204811, 12.0219307
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6816101, 14.6893196
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1180801, 15.1337967
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6058807, 13.6117630
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1074486, 12.1080742
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9693146, 12.9566803
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9483414, 20.9378738
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0168076, 15.0145073
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6181183, 16.6063766
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7014465, 26.6872253
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7810745, 14.7786903
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2094345, 17.1999588
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4150963, 14.4221992
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5171814, 14.5227127
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5953217, 12.6000481
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9323273, 14.9335670
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2889099, 14.2846375
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2246246, 9.2290993
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7840157, 13.7842331
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2401543, 19.2353363
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.1954803, 13.2031288
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9164162, 14.9110680
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5563583, 13.5564880
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6324081, 14.6423683
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2804146, 13.2879028
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5408058, 16.5571556
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0604019, 14.0833206
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9323807, 12.9474335
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0243378, 15.0335770
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9057693, 18.9068680
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.1946754, 18.2209091
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -15.9943619, 16.0074234
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3914413, 14.3961601
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6747360, 16.6850510
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3309174, 17.3281860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 955

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5432529, upper bound: 12.4564288
time: 9.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5625122, upper bound: 12.4371578
time: 10.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8583336, 13.8541565
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4856453, 8.4889736
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4619598, 13.4607468
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0205917, 12.0201302
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6816711, 14.6892395
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1191711, 15.1318207
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6061707, 13.6106033
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1077309, 12.1066132
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9682655, 12.9571114
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9485092, 20.9378738
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0164337, 15.0145454
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6140366, 16.6082611
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7012787, 26.6874695
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7818527, 14.7771378
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2094727, 17.2003937
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4152451, 14.4219570
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5172272, 14.5227394
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5953865, 12.6000671
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9314117, 14.9340668
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2888031, 14.2848053
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2245026, 9.2291412
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7843971, 13.7837372
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2383232, 19.2369232
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.1954803, 13.2031212
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9155312, 14.9116936
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5568695, 13.5561371
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6331749, 14.6405640
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2804298, 13.2878952
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5408058, 16.5571518
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0614395, 14.0814629
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9324150, 12.9473648
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0224915, 15.0347633
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9021606, 18.9070702
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.1948586, 18.2209320
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -15.9929123, 16.0077667
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3910141, 14.3963909
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6748123, 16.6851845
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3313751, 17.3276215

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 955

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5409971, upper bound: 12.4581235
time: 11.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5602652, upper bound: 12.4388524
time: 7.84 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 21.48 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 21.48
Output dim: 14, lower bound: -12.4388525, upper bound: 12.5602652
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 21.48
Output dim: 14, lower bound: -12.4581235, upper bound: 12.5409971
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 21.48
Output dim: 14, lower bound: -12.4371578, upper bound: 12.5625121
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 21.48
Output dim: 14, lower bound: -12.4564288, upper bound: 12.5432529
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 21.48
Output dim: 14, lower bound: -12.5432529, upper bound: 12.4564288
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 21.48
Output dim: 14, lower bound: -12.5625122, upper bound: 12.4371578
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 21.48
Output dim: 14, lower bound: -12.5409971, upper bound: 12.4581235
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 21.48
Output dim: 14, lower bound: -12.5602652, upper bound: 12.4388524

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8550797, 13.8600082
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4889526, 8.4844704
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4539680, 13.4565201
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0200005, 12.0209770
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6897736, 14.6805077
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9183731, 15.9181862
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1276474, 15.1120720
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6080475, 13.6041603
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1124229, 12.1156960
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9612541, 12.9717064
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9397507, 20.9510117
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -14.9953575, 14.9947243
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6270485, 16.6304398
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.6911774, 26.7072678
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7805252, 14.7845879
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.1889038, 17.1997948
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4068661, 14.4013329
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5226860, 14.5171623
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5988579, 12.5943413
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9343185, 14.9313202
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2735519, 14.2805023
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2234116, 9.2195473
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7799416, 13.7803497
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2412529, 19.2453156
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2139053, 13.2085457
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9110374, 14.9158134
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5538483, 13.5539970
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6341972, 14.6266747
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2816811, 13.2728577
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5524216, 16.5313606
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0787086, 14.0539703
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9426308, 12.9233093
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0305328, 15.0173950
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9016418, 18.8941422
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2271767, 18.1975174
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -15.9958687, 15.9759445
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3907814, 14.3852787
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6867638, 16.6757088
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3293839, 17.3336029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1589

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4371163, upper bound: 12.5498351
time: 28.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4270322, upper bound: 12.5624720
time: 20.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8590164, 13.8550072
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4842339, 8.4897690
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4560051, 13.4539680
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0204391, 12.0216980
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6805038, 14.6883812
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9178848, 15.9192619
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1120682, 15.1282654
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6036682, 13.6089172
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1146736, 12.1136055
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9717941, 12.9612541
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9497070, 20.9395981
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -14.9950638, 14.9952278
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6326485, 16.6251450
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7061310, 26.6911697
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7831802, 14.7813034
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.1985550, 17.1884193
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4013309, 14.4067307
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5170860, 14.5221863
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5942574, 12.5981407
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9317474, 14.9329643
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2792969, 14.2735519
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2196274, 9.2228699
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7803535, 13.7800446
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2455711, 19.2407379
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2085495, 13.2132759
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9160881, 14.9106827
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5539589, 13.5536842
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6266747, 14.6349030
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2728577, 13.2811127
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5313644, 16.5499306
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0539742, 14.0776138
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9233131, 12.9399872
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0180588, 15.0279541
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8975525, 18.8999214
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.1973076, 18.2239532
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -15.9770470, 15.9925041
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3854675, 14.3908501
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6754951, 16.6860352
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3321838, 17.3294830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1589

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5624720, upper bound: 12.4270321
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5523963, upper bound: 12.4371163
time: 6.73 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 15.66 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 15.66
Output dim: 14, lower bound: -12.4371163, upper bound: 12.5498351
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 15.66
Output dim: 14, lower bound: -12.4270322, upper bound: 12.5624720
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 15.66
Output dim: 14, lower bound: -12.5624720, upper bound: 12.4270321
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 15.66
Output dim: 14, lower bound: -12.5523963, upper bound: 12.4371163

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8386345, 13.8454170
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4880714, 8.4864464
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4403534, 13.4444771
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0304756, 12.0299492
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6733475, 14.6675606
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0948029, 15.0749855
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.5932770, 13.5923004
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1166916, 12.1208649
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9622307, 12.9728279
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9122925, 20.9192657
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -14.9700165, 14.9657440
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6257248, 16.6291504
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.6918716, 26.7080307
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7795258, 14.7862358
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.1958389, 17.2059441
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.3987236, 14.3954468
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5033417, 14.5005493
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5978546, 12.5934029
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9391174, 14.9359322
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2699318, 14.2773056
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2134323, 9.2117386
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7718048, 13.7735023
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2355270, 19.2387695
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2113953, 13.2090340
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9079132, 14.9100342
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5400429, 13.5394058
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6015854, 14.5980568
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2735214, 13.2628174
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5531654, 16.5321083
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0638084, 14.0342941
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9431305, 12.9238205
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0306053, 15.0174561
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8988266, 18.8906479
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2242508, 18.1951218
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -15.9902344, 15.9737129
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3772202, 14.3700867
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6767273, 16.6639137
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2965012, 17.2967453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1728

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4224070, upper bound: 12.5623494
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4266129, upper bound: 12.5536181
time: 5.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8444252, 13.8385620
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4862061, 8.4888954
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4439545, 13.4403496
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0294075, 12.0321732
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6675644, 14.6719551
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0749855, 15.0954208
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.5918121, 13.5941391
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1198425, 12.1178741
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9729118, 12.9622307
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9179688, 20.9121323
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -14.9660873, 14.9698830
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6313553, 16.6238251
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7068863, 26.6918640
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7848282, 14.7803078
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2047043, 17.1953545
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.3954430, 14.3985939
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5004730, 14.5028267
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5933189, 12.5971413
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9363632, 14.9377518
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2760963, 14.2699356
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2118225, 9.2128868
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7734985, 13.7719193
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2390289, 19.2350006
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2090340, 13.2107697
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9103012, 14.9075699
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5393715, 13.5398827
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.5980568, 14.6022720
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2628174, 13.2729759
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5321083, 16.5506821
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0342903, 14.0627136
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9238205, 12.9404793
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0181236, 15.0280228
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8940659, 18.8971062
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.1949158, 18.2210159
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -15.9748230, 15.9868774
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3702774, 14.3772812
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6637039, 16.6759987
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2953339, 17.2965927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1728

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5536181, upper bound: 12.4266128
time: 7.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5623494, upper bound: 12.4224070
time: 6.79 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 17.01 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 17.01
Output dim: 14, lower bound: -12.4224070, upper bound: 12.5623494
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 17.01
Output dim: 14, lower bound: -12.4266129, upper bound: 12.5536181
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 17.01
Output dim: 14, lower bound: -12.5536181, upper bound: 12.4266128
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 17.01
Output dim: 14, lower bound: -12.5623494, upper bound: 12.4224070

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8385925, 13.8453903
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4884949, 8.4862556
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4402008, 13.4445648
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0300636, 12.0295849
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6720734, 14.6654243
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0917778, 15.0705185
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.5932388, 13.5918617
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1147003, 12.1195316
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9615517, 12.9723358
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9100876, 20.9184952
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -14.9690247, 14.9643402
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6232758, 16.6254845
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.6898270, 26.7066574
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7794228, 14.7862053
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.1939087, 17.2046356
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.3982391, 14.3947697
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5033035, 14.5001945
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5977364, 12.5930138
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9391518, 14.9358826
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2678413, 14.2760773
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2129631, 9.2112045
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7717819, 13.7734795
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2343979, 19.2383804
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2115097, 13.2090378
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9065170, 14.9091644
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5400581, 13.5394058
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6017456, 14.5977287
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2719116, 13.2604370
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5488815, 16.5257378
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0619926, 14.0315247
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9394722, 12.9185104
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0275040, 15.0127716
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8974304, 18.8884583
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2184792, 18.1865540
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -15.9843597, 15.9649811
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3766708, 14.3695869
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6753540, 16.6618919
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2962341, 17.2970810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 977

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4217200, upper bound: 12.5527049
time: 7.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4148592, upper bound: 12.5616858
time: 7.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8443947, 13.8385239
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4860191, 8.4893188
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4440384, 13.4401970
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0290413, 12.0317631
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6654282, 14.6706848
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0705185, 15.0923920
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.5913773, 13.5941048
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1185074, 12.1158848
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9724197, 12.9615517
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9171829, 20.9099350
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -14.9646759, 14.9688988
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6276855, 16.6213760
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7055283, 26.6898193
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7848015, 14.7802048
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2033920, 17.1934204
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.3947716, 14.3981152
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5001144, 14.5027962
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5929260, 12.5970268
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9363060, 14.9377975
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2748909, 14.2678452
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2112885, 9.2124176
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7734756, 13.7719002
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2386322, 19.2338638
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2090378, 13.2108841
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9094315, 14.9061737
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5393677, 13.5398979
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.5977287, 14.6024399
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2604370, 13.2713623
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5257339, 16.5463943
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0315208, 14.0609055
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9185066, 12.9368172
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0134354, 15.0249329
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8918686, 18.8957214
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.1863441, 18.2152557
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -15.9660873, 15.9809990
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3697777, 14.3767300
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6616745, 16.6746292
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2956696, 17.2963409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 977

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5616859, upper bound: 12.4148592
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5527050, upper bound: 12.4217200
time: 14.82 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 23.06 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 23.06
Output dim: 14, lower bound: -12.4217200, upper bound: 12.5527049
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 12, time: 23.06
Output dim: 14, lower bound: -12.4148592, upper bound: 12.5616858
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 12, time: 23.06
Output dim: 14, lower bound: -12.5616859, upper bound: 12.4148592
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 23.06
Output dim: 14, lower bound: -12.5527050, upper bound: 12.4217200

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8356247, 13.8427620
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4880981, 8.4858799
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4379196, 13.4426041
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0259628, 12.0259342
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6756058, 14.6680794
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0866318, 15.0647507
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.5910492, 13.5899544
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1077118, 12.1132870
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9617882, 12.9726067
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.8992310, 20.9086914
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -14.9657822, 14.9605598
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6206474, 16.6210175
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.6840515, 26.7017212
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7733612, 14.7806778
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.1813240, 17.1935959
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.3983078, 14.3948345
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5006409, 14.4972572
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5952682, 12.5902252
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9348221, 14.9307442
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2689056, 14.2778625
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2118950, 9.2100639
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7720032, 13.7736855
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2318649, 19.2356567
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2117615, 13.2090721
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9064941, 14.9091148
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5394630, 13.5392609
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6005859, 14.5964470
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2639694, 13.2513733
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5283737, 16.5022888
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0518036, 14.0198708
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9214172, 12.8979034
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0070190, 14.9894066
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8827972, 18.8716888
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.1938705, 18.1590271
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -15.9554710, 15.9320183
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3756180, 14.3684998
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6677895, 16.6532249
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2982368, 17.2995224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2036
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 586

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4101459, upper bound: 12.5588401
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4120110, upper bound: 12.5569830
time: 8.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8417702, 13.8355598
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4856415, 8.4889126
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4420776, 13.4379196
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0254059, 12.0276546
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6680832, 14.6742096
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0647507, 15.0870934
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.5894623, 13.5919151
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1122742, 12.1088963
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9726944, 12.9617882
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9074173, 20.8990707
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -14.9608955, 14.9656410
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6232262, 16.6187096
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7005920, 26.6840515
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7792664, 14.7741432
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.1923752, 17.1808434
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.3948364, 14.3981781
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.4971771, 14.5001373
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5901413, 12.5945663
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9311752, 14.9334717
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2766571, 14.2688980
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2101440, 9.2113571
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7736816, 13.7721138
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2358932, 19.2313461
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2090721, 13.2111397
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9093781, 14.9061508
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5392227, 13.5392952
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.5964432, 14.6012726
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2513733, 13.2634163
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5022888, 16.5258789
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0198669, 14.0507202
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.8979034, 12.9187813
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -14.9900742, 15.0044518
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.8751068, 18.8810921
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.1588211, 18.1906509
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -15.9331245, 15.9521065
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3686905, 14.3756828
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6530190, 16.6670532
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2981071, 17.2983246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2036
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 586

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5569830, upper bound: 12.4120110
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5588402, upper bound: 12.4101458
time: 8.79 seconds

## Summary of splitting (split count: 12)
- Time for RS candidates: 17.68 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 13, time: 17.68
Output dim: 14, lower bound: -12.4101459, upper bound: 12.5588401
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 13, time: 17.68
Output dim: 14, lower bound: -12.4120110, upper bound: 12.5569830
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 13, time: 17.68
Output dim: 14, lower bound: -12.5569830, upper bound: 12.4120110
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 13, time: 17.68
Output dim: 14, lower bound: -12.5588402, upper bound: 12.4101458

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 23.23 + 874.76 = 897.98 seconds
