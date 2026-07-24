## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 3600 seconds
Split limit: 100
Threshold: 37.4874905906


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341)
1: (-23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751)
2: (-18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7670898, 50.7670937)
3: (-19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7754364, 53.7754440)
4: (-23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012)
5: (-21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0920868, 56.0921021)
6: (-42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501)
7: (-30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0413132, 64.0413132)
8: (-29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192)
9: (-24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9636116, 54.9636154)
10: (-45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318)
11: (-48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746)
12: (-52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4354858, 69.4354935)
13: (-35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266)
14: (-78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879)
15: (-30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065)
16: (-46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2572174, 77.2572327)
17: (-77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000)
18: (-45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464)
19: (-34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724)
20: (-30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395)
21: (-42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793)
22: (-43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864)
23: (-34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452)
24: (-36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056)
25: (-35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590)
26: (-53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457)
27: (-36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078)
28: (-33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381)
29: (-45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077)
30: (-42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827)
31: (-42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359)
32: (-38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531)
33: (-48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046)
34: (-47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2930145, 68.2930222)
35: (-41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7198257, 67.7198334)
36: (-42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2822266, 68.2822189)
37: (-66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7244415, 86.7244415)
38: (-52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0830994, 82.0831070)
39: (-60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865)
40: (-53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266)
41: (-39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605)
42: (-32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.75 + 76.79 = 79.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -37.5438063, upper bound: 37.5438063

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 932

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 517

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5398761, upper bound: 37.5396858
time: 63.16 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5396858, upper bound: 37.5398762
time: 49.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 112.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 112.68
Output dim: 8, lower bound: -37.5398761, upper bound: 37.5396858
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 112.68
Output dim: 8, lower bound: -37.5396858, upper bound: 37.5398762

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7671204, 50.7670937
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7756195, 53.7753525
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0921936, 56.0920601
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0414276, 64.0412521
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9639130, 54.9635048
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4353638, 69.4357910
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2572479, 77.2572021
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2930298, 68.2930450
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7197800, 67.7199554
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2821503, 68.2823944
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7242737, 86.7248306
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0829773, 82.0834198
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 630

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5336454, upper bound: 37.5285812
time: 64.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5287621, upper bound: 37.5334639
time: 51.02 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7670898, 50.7670937
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7753525, 53.7754440
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0920563, 56.0921021
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0412598, 64.0413132
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9635010, 54.9636154
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4354858, 69.4353714
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2571869, 77.2572327
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2930145, 68.2930145
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7198257, 67.7197800
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2822266, 68.2821503
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7244415, 86.7242737
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0830994, 82.0829849
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 782

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 764

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5369414, upper bound: 37.5365496
time: 55.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5363601, upper bound: 37.5371310
time: 50.98 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 108.93 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 108.93
Output dim: 8, lower bound: -37.5336454, upper bound: 37.5285812
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 108.93
Output dim: 8, lower bound: -37.5287621, upper bound: 37.5334639
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 108.93
Output dim: 8, lower bound: -37.5369414, upper bound: 37.5365496
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 108.93
Output dim: 8, lower bound: -37.5363601, upper bound: 37.5371310

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7666016, 50.7666588
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7711411, 53.7716103
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0917511, 56.0919876
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0401611, 64.0404587
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9475021, 54.9482193
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4353943, 69.4346771
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2561798, 77.2562485
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2921600, 68.2920990
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7128754, 67.7125549
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2728119, 68.2723999
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7036743, 86.7027283
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0663071, 82.0655670
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1600

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5283382, upper bound: 37.5279904
time: 55.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5330544, upper bound: 37.5232728
time: 47.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7666931, 50.7665672
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7718735, 53.7708817
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0921173, 56.0916176
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0406494, 64.0399857
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9486237, 54.9470978
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4342346, 69.4358215
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2563019, 77.2561340
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2920685, 68.2921753
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7123718, 67.7130508
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2721558, 68.2730637
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7021637, 86.7042389
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0651321, 82.0667572
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 513

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1659

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5264556, upper bound: 37.5284832
time: 57.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5237462, upper bound: 37.5311311
time: 56.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7670822, 50.7671852
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7753296, 53.7762070
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0920563, 56.0924835
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0412292, 64.0418091
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9634628, 54.9647865
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4362411, 69.4348755
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2572021, 77.2573547
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2930908, 68.2930069
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7203369, 67.7197571
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2828522, 68.2820740
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7260208, 86.7242432
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0840454, 82.0826416
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1549

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1668

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5351464, upper bound: 37.5357111
time: 54.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5361027, upper bound: 37.5347540
time: 63.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7670898, 50.7670898
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7753525, 53.7754250
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0920563, 56.0920906
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0412598, 64.0412979
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9635010, 54.9635849
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4349899, 69.4353714
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2571869, 77.2572250
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2930145, 68.2930145
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7198181, 67.7197800
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2821350, 68.2821503
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7244034, 86.7242737
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0827637, 82.0829849
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1674

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 899

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5324579, upper bound: 37.5369123
time: 53.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5361416, upper bound: 37.5332300
time: 56.11 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 111.72 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 111.72
Output dim: 8, lower bound: -37.5283382, upper bound: 37.5279904
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 111.72
Output dim: 8, lower bound: -37.5330544, upper bound: 37.5232728
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 111.72
Output dim: 8, lower bound: -37.5264556, upper bound: 37.5284832
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 111.72
Output dim: 8, lower bound: -37.5237462, upper bound: 37.5311311
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 111.72
Output dim: 8, lower bound: -37.5351464, upper bound: 37.5357111
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 111.72
Output dim: 8, lower bound: -37.5361027, upper bound: 37.5347540
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 111.72
Output dim: 8, lower bound: -37.5324579, upper bound: 37.5369123
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 111.72
Output dim: 8, lower bound: -37.5361416, upper bound: 37.5332300

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7663879, 50.7664719
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7694550, 53.7701569
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0908813, 56.0912361
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0390167, 64.0394669
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9447861, 54.9458771
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4339066, 69.4327927
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2559052, 77.2560120
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2919922, 68.2919235
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7118225, 67.7113647
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2714310, 68.2707825
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7005157, 86.6990509
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0638275, 82.0626831
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1701

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 635

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5147256, upper bound: 37.5174731
time: 61.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5178398, upper bound: 37.5143561
time: 56.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7664185, 50.7664452
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7696991, 53.7699165
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0910034, 56.0911140
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0391693, 64.0393143
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9451599, 54.9455032
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4335251, 69.4331818
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2559357, 77.2559738
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2919769, 68.2919464
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7116699, 67.7115250
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2712021, 68.2710037
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.6999969, 86.6995544
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0634308, 82.0630798
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 705

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 738

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5072962, upper bound: 37.4857222
time: 49.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4955241, upper bound: 37.4974934
time: 61.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7665176, 50.7666397
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7707901, 53.7717743
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0914764, 56.0919762
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0400391, 64.0406723
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9428406, 54.9443626
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4333344, 69.4317780
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2558746, 77.2560349
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2922363, 68.2921219
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7133102, 67.7126541
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2724457, 68.2715607
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7050781, 86.7030411
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0673065, 82.0657120
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 982

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 683

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5144573, upper bound: 37.5205381
time: 55.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5184647, upper bound: 37.5166579
time: 57.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7667618, 50.7663918
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7727737, 53.7697945
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0924835, 56.0909805
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0413208, 64.0393829
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9458923, 54.9413185
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4301910, 69.4349136
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2562103, 77.2557068
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2920227, 68.2923279
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7119827, 67.7139893
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2706451, 68.2733612
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7009735, 86.7071381
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0640869, 82.0689240
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1674

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 614

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5180855, upper bound: 37.5258755
time: 60.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5180855, upper bound: 37.5259590
time: 62.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7662964, 50.7662468
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7710571, 53.7706451
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0891876, 56.0889740
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0372620, 64.0369949
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9531441, 54.9525146
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4275894, 69.4282303
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2561035, 77.2560425
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2922668, 68.2923050
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7149658, 67.7152481
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2750244, 68.2754135
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7094879, 86.7103729
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0710678, 82.0717316
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1659

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4801879, upper bound: 37.5316633
time: 55.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5311030, upper bound: 37.4807693
time: 81.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7661438, 50.7664070
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7697754, 53.7719193
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0885468, 56.0896149
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0364227, 64.0378265
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9511833, 54.9544754
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4296036, 69.4262085
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2559204, 77.2562485
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2924042, 68.2921753
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7158203, 67.7143860
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2761841, 68.2742538
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7121429, 86.7077332
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0731430, 82.0696564
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1648

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 915

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5356824, upper bound: 37.5331193
time: 69.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5338982, upper bound: 37.5343398
time: 372.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7662506, 50.7663689
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7684174, 53.7695312
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0885696, 56.0891190
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0370178, 64.0377426
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9528275, 54.9545097
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4261093, 69.4248276
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2561188, 77.2563171
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2924042, 68.2922897
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7158356, 67.7151031
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2768707, 68.2759247
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7122116, 86.7099228
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0731888, 82.0717010
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 705

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 614

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5245699, upper bound: 37.5290568
time: 52.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5245699, upper bound: 37.5290568
time: 56.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7663727, 50.7662392
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7694626, 53.7684937
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0891037, 56.0885963
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0376892, 64.0370712
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9544296, 54.9529076
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4244614, 69.4264755
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2562866, 77.2561493
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2922821, 68.2923965
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7151337, 67.7158051
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2759247, 68.2768707
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7100449, 86.7120743
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0714798, 82.0733948
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 706

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1664

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5356806, upper bound: 37.5330459
time: 52.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5356806, upper bound: 37.5327696
time: 53.69 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 108.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 108.00
Output dim: 8, lower bound: -37.5147256, upper bound: 37.5174731
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 108.00
Output dim: 8, lower bound: -37.5178398, upper bound: 37.5143561
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 108.00
Output dim: 8, lower bound: -37.5072962, upper bound: 37.4857222
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 108.00
Output dim: 8, lower bound: -37.4955241, upper bound: 37.4974934
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 108.00
Output dim: 8, lower bound: -37.5144573, upper bound: 37.5205381
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 108.00
Output dim: 8, lower bound: -37.5184647, upper bound: 37.5166579
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 108.00
Output dim: 8, lower bound: -37.5180855, upper bound: 37.5258755
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 108.00
Output dim: 8, lower bound: -37.5180855, upper bound: 37.5259590
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 108.00
Output dim: 8, lower bound: -37.4801879, upper bound: 37.5316633
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 108.00
Output dim: 8, lower bound: -37.5311030, upper bound: 37.4807693
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 108.00
Output dim: 8, lower bound: -37.5356824, upper bound: 37.5331193
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 108.00
Output dim: 8, lower bound: -37.5338982, upper bound: 37.5343398
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 108.00
Output dim: 8, lower bound: -37.5245699, upper bound: 37.5290568
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 108.00
Output dim: 8, lower bound: -37.5245699, upper bound: 37.5290568
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 108.00
Output dim: 8, lower bound: -37.5356806, upper bound: 37.5330459
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 108.00
Output dim: 8, lower bound: -37.5356806, upper bound: 37.5327696

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7659531, 50.7658920
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7656708, 53.7652245
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0889740, 56.0887413
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0365753, 64.0362854
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9423447, 54.9416504
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4287109, 69.4294205
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2555847, 77.2555084
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2914734, 68.2915192
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7084198, 67.7087402
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2672577, 68.2676697
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.6900711, 86.6910095
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0558319, 82.0565796
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1550

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 634

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5099529, upper bound: 37.5139678
time: 52.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5086890, upper bound: 37.5142288
time: 54.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7658005, 50.7660370
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7645111, 53.7663765
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0883942, 56.0893250
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0358276, 64.0370407
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9405670, 54.9434319
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4305420, 69.4275894
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2554016, 77.2557068
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2915955, 68.2913971
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7092133, 67.7079544
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2683105, 68.2666092
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.6924667, 86.6886139
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0577240, 82.0547028
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 932

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 824

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5121212, upper bound: 37.5142955
time: 58.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5177782, upper bound: 37.5086286
time: 57.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7664185, 50.7664146
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7696533, 53.7696266
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0909882, 56.0909805
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0391693, 64.0391464
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9451447, 54.9451103
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4330597, 69.4330902
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2559357, 77.2559280
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2919464, 68.2919464
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7114944, 67.7115097
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2709656, 68.2710037
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.6995087, 86.6995544
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0630341, 82.0630722
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 841

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 705

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4717054, upper bound: 37.4399931
time: 74.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4615849, upper bound: 37.4501123
time: 91.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7664185, 50.7664452
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7696991, 53.7698708
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0910034, 56.0911026
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0391693, 64.0393066
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9451599, 54.9454880
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4334412, 69.4331818
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2559357, 77.2559662
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2919617, 68.2919464
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7116623, 67.7115250
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2712097, 68.2710037
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7000122, 86.6995544
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0634308, 82.0630798
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 752

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 966

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4829314, upper bound: 37.4974176
time: 34.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4954472, upper bound: 37.4848798
time: 48.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7664948, 50.7666206
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7706451, 53.7716446
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0914001, 56.0919037
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0399323, 64.0405884
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9426804, 54.9442368
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4331436, 69.4315491
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2558594, 77.2560272
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2922211, 68.2921143
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7132339, 67.7125626
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2723312, 68.2714157
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7047882, 86.7027206
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0671387, 82.0655060
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1655

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 823

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5051267, upper bound: 37.5202858
time: 84.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5143843, upper bound: 37.5110284
time: 54.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7664948, 50.7666206
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7706604, 53.7716293
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0914154, 56.0918961
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0399475, 64.0405731
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9427109, 54.9442062
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4331131, 69.4315796
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2558594, 77.2560196
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2922211, 68.2921219
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7132187, 67.7125778
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2723160, 68.2714310
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7047577, 86.7027588
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0671082, 82.0655365
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1657

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1568

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5178418, upper bound: 37.5160519
time: 53.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5178415, upper bound: 37.5160521
time: 60.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7661667, 50.7659073
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7667084, 53.7646408
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0901947, 56.0891609
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0381622, 64.0368042
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9384766, 54.9352989
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4271851, 69.4304657
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2553101, 77.2549744
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2914429, 68.2916565
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7071075, 67.7085114
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2640610, 68.2659531
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.6873627, 86.6916428
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0523224, 82.0557022
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 706

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4965234, upper bound: 37.4801503
time: 67.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4722686, upper bound: 37.5044103
time: 58.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7663727, 50.7657967
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7683182, 53.7637329
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0910187, 56.0887032
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0392151, 64.0362167
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9409561, 54.9339027
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4257507, 69.4330139
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2555695, 77.2548218
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2913513, 68.2918243
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7064972, 67.7095947
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2632370, 68.2674179
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.6854858, 86.6949768
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0508575, 82.0583191
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1638

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1016

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4648782, upper bound: 37.5257239
time: 49.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5178503, upper bound: 37.4729719
time: 53.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7661514, 50.7660866
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7699585, 53.7694817
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0886841, 56.0884361
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0367584, 64.0364380
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9530144, 54.9522667
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4256439, 69.4264374
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2560730, 77.2559738
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2922668, 68.2923050
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7148590, 67.7151718
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2745056, 68.2749634
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7080536, 86.7090988
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0699005, 82.0707016
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 706

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1608

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4784060, upper bound: 37.5303192
time: 55.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4788300, upper bound: 37.5299024
time: 51.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7662964, 50.7660980
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7710571, 53.7695580
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0891876, 56.0884781
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0372620, 64.0364914
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9531441, 54.9523849
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4257660, 69.4282303
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2561035, 77.2559891
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2922668, 68.2923050
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7148895, 67.7152481
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2745972, 68.2754135
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7082062, 86.7103729
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0700531, 82.0717316
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 649

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5298891, upper bound: 37.4366008
time: 49.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4870857, upper bound: 37.4795649
time: 62.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7651367, 50.7655640
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7625275, 53.7659035
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0844269, 56.0861206
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0312805, 64.0334702
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9386253, 54.9437828
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4197617, 69.4144516
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2546387, 77.2552032
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2917023, 68.2913437
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7112579, 67.7089920
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2709579, 68.2678909
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.6977463, 86.6908112
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0618591, 82.0563660
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1652

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 778

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5354518, upper bound: 37.5304941
time: 48.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5330632, upper bound: 37.5328877
time: 52.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7652893, 50.7654114
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7637558, 53.7646790
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0850525, 56.0855026
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0320740, 64.0326691
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9405098, 54.9418983
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4178391, 69.4163895
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2548523, 77.2550049
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2915649, 68.2914658
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7104187, 67.7098160
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2698441, 68.2690048
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.6952133, 86.6933441
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0598755, 82.0583649
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 861

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 512

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5324760, upper bound: 37.5339801
time: 57.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5341202, upper bound: 37.5323344
time: 57.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7656555, 50.7658997
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7623596, 53.7643890
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0862808, 56.0873070
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0338593, 64.0351791
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9454041, 54.9485016
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4227448, 69.4200439
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2552338, 77.2555695
17: -77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000
18: -45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464
19: -34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724
20: -30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395
21: -42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793
22: -43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864
23: -34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452
24: -36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056
25: -35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590
26: -53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457
27: -36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078
28: -33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381
29: -45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077
30: -42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827
31: -42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359
32: -38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531
33: -48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2918243, 68.2916183
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7109528, 67.7096100
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2702789, 68.2685165
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.6985931, 86.6944275
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0614166, 82.0584717
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 680

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1632

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5206573, upper bound: 37.5284140
time: 49.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5239380, upper bound: 37.5251266
time: 56.74 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 107.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.5099529, upper bound: 37.5139678
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.5086890, upper bound: 37.5142288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.5121212, upper bound: 37.5142955
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.5177782, upper bound: 37.5086286
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.4717054, upper bound: 37.4399931
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.4615849, upper bound: 37.4501123
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.4829314, upper bound: 37.4974176
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.4954472, upper bound: 37.4848798
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.5051267, upper bound: 37.5202858
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.5143843, upper bound: 37.5110284
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.5178418, upper bound: 37.5160519
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.5178415, upper bound: 37.5160521
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.4965234, upper bound: 37.4801503
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.4722686, upper bound: 37.5044103
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.4648782, upper bound: 37.5257239
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.5178503, upper bound: 37.4729719
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.4784060, upper bound: 37.5303192
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.4788300, upper bound: 37.5299024
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.5298891, upper bound: 37.4366008
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.4870857, upper bound: 37.4795649
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.5354518, upper bound: 37.5304941
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.5330632, upper bound: 37.5328877
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.5324760, upper bound: 37.5339801
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.5341202, upper bound: 37.5323344
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.5206573, upper bound: 37.5284140
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 107.94
Output dim: 8, lower bound: -37.5239380, upper bound: 37.5251266
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 107.94
Output dim: 8, lower bound: -37.5245699, upper bound: 37.5290568
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 107.94
Output dim: 8, lower bound: -37.5356806, upper bound: 37.5330459
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 107.94
Output dim: 8, lower bound: -37.5356806, upper bound: 37.5327696

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 79.53 + 3592.36 = 3671.89 seconds
