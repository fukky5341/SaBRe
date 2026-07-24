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
execution time: IAR + RelationalAnalysis = 2.88 + 78.11 = 80.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -37.5438063, upper bound: 37.5438063

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 822

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5092256, upper bound: 37.5356623
time: 51.77 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5356623, upper bound: 37.5092256
time: 61.66 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 113.56 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 113.56
Output dim: 8, lower bound: -37.5092256, upper bound: 37.5356623
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 113.56
Output dim: 8, lower bound: -37.5356623, upper bound: 37.5092256

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7662735, 50.7672119
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7697144, 53.7772331
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0887756, 56.0925598
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0370102, 64.0419159
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9534721, 54.9650345
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4369507, 69.4250488
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2564850, 77.2577057
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
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2937088, 68.2929382
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7229462, 67.7178802
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2850037, 68.2781601
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7339020, 86.7183380
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0906372, 82.0784302
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 822

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4600154, upper bound: 37.4880515
time: 57.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4614304, upper bound: 37.4866370
time: 61.60 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7670898, 50.7662735
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7754364, 53.7697144
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0920868, 56.0887756
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0413132, 64.0370102
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9636116, 54.9534760
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4250488, 69.4354935
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2572174, 77.2564774
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
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2929459, 68.2930222
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7178802, 67.7198334
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2781677, 68.2822189
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7183380, 86.7244415
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0784302, 82.0831070
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 822

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4866370, upper bound: 37.4614305
time: 58.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4880514, upper bound: 37.4600154
time: 66.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 127.56 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 127.56
Output dim: 8, lower bound: -37.4600154, upper bound: 37.4880515
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 127.56
Output dim: 8, lower bound: -37.4614304, upper bound: 37.4866370
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 127.56
Output dim: 8, lower bound: -37.4866370, upper bound: 37.4614305
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 127.56
Output dim: 8, lower bound: -37.4880514, upper bound: 37.4600154

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7661743, 50.7677689
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7690506, 53.7818069
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0884247, 56.0948486
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0365524, 64.0448761
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9530067, 54.9726181
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4441071, 69.4239273
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2563782, 77.2584534
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
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2942276, 68.2929077
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7264175, 67.7178116
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2896423, 68.2780228
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7441788, 86.7177734
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0987091, 82.0779877
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 822

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1722

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4572094, upper bound: 37.4742359
time: 63.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4462030, upper bound: 37.4852123
time: 55.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341
1: -23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751
2: -18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7670898, 50.7661743
3: -19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7754364, 53.7690506
4: -23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012
5: -21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0920868, 56.0884247
6: -42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501
7: -30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0413132, 64.0365524
8: -29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192
9: -24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9636116, 54.9530067
10: -45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318
11: -48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746
12: -52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4239197, 69.4354935
13: -35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266
14: -78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879
15: -30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065
16: -46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2572174, 77.2563782
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
34: -47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2929153, 68.2930222
35: -41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7178116, 67.7198334
36: -42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2780151, 68.2822189
37: -66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7177811, 86.7244415
38: -52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0779877, 82.0831070
39: -60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865
40: -53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266
41: -39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605
42: -32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 822

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1722

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4852123, upper bound: 37.4462030
time: 66.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4742359, upper bound: 37.4572094
time: 55.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 124.09 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 124.09
Output dim: 8, lower bound: -37.4572094, upper bound: 37.4742359
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 124.09
Output dim: 8, lower bound: -37.4462030, upper bound: 37.4852123
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 124.09
Output dim: 8, lower bound: -37.4852123, upper bound: 37.4462030
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 124.09
Output dim: 8, lower bound: -37.4742359, upper bound: 37.4572094

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 80.99 + 607.95 = 688.94 seconds
