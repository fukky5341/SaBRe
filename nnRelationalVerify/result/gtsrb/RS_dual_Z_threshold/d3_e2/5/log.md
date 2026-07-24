## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 5)
Time budget: 7200 seconds
Split limit: 100
Threshold: 38.9746791072


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328)
1: (-31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664)
2: (-30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690)
3: (-34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942)
4: (-40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316)
5: (-37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755)
6: (-56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726)
7: (-43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831)
8: (-39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809)
9: (-34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275)
10: (-55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739)
11: (-56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778)
12: (-59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474)
13: (-48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827)
14: (-81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586)
15: (-40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336)
16: (-58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498)
17: (-85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696)
18: (-49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788)
19: (-41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662)
20: (-35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226)
21: (-49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371)
22: (-51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297)
23: (-39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429)
24: (-45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464)
25: (-38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314)
26: (-59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385)
27: (-49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097)
28: (-37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770)
29: (-55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838)
30: (-47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052)
31: (-49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653)
32: (-49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205)
33: (-72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273)
34: (-61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420)
35: (-57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084)
36: (-57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705)
37: (-85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175)
38: (-69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412)
39: (-85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074)
40: (-75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985)
41: (-54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703)
42: (-39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.87 + 101.35 = 104.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -39.0136928, upper bound: 39.0136928

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0116519, upper bound: 38.9017386
time: 70.40 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9017386, upper bound: 39.0116519
time: 71.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 142.28 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 142.28
Output dim: 2, lower bound: -39.0116519, upper bound: 38.9017386
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 142.28
Output dim: 2, lower bound: -38.9017386, upper bound: 39.0116519

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9721702, upper bound: 38.9003980
time: 111.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0103065, upper bound: 38.8622405
time: 76.22 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8622405, upper bound: 39.0103065
time: 99.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9003980, upper bound: 38.9721702
time: 81.80 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 184.12 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 184.12
Output dim: 2, lower bound: -38.9721702, upper bound: 38.9003980
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 184.12
Output dim: 2, lower bound: -39.0103065, upper bound: 38.8622405
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 184.12
Output dim: 2, lower bound: -38.8622405, upper bound: 39.0103065
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 184.12
Output dim: 2, lower bound: -38.9003980, upper bound: 38.9721702

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 663

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0081153, upper bound: 38.7906293
time: 73.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9189900, upper bound: 38.8571891
time: 85.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 663

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8571891, upper bound: 38.9189900
time: 70.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7906293, upper bound: 39.0081153
time: 78.91 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 151.65 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 151.65
Output dim: 2, lower bound: -39.0081153, upper bound: 38.7906293
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 151.65
Output dim: 2, lower bound: -38.9189900, upper bound: 38.8571891
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 151.65
Output dim: 2, lower bound: -38.8571891, upper bound: 38.9189900
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 151.65
Output dim: 2, lower bound: -38.7906293, upper bound: 39.0081153

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 648

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0065803, upper bound: 38.6793751
time: 77.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8714502, upper bound: 38.7888930
time: 81.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 648

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7888930, upper bound: 38.8714502
time: 90.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6793751, upper bound: 39.0065803
time: 69.22 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 162.25 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 162.25
Output dim: 2, lower bound: -39.0065803, upper bound: 38.6793751
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 162.25
Output dim: 2, lower bound: -38.8714502, upper bound: 38.7888930
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 162.25
Output dim: 2, lower bound: -38.7888930, upper bound: 38.8714502
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 162.25
Output dim: 2, lower bound: -38.6793751, upper bound: 39.0065803

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 665

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0040709, upper bound: 38.6300629
time: 69.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9460297, upper bound: 38.6729371
time: 78.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 665

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6729371, upper bound: 38.9460297
time: 65.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6300629, upper bound: 39.0040709
time: 67.91 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 135.96 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 135.96
Output dim: 2, lower bound: -39.0040709, upper bound: 38.6300629
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 135.96
Output dim: 2, lower bound: -38.9460297, upper bound: 38.6729371
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 135.96
Output dim: 2, lower bound: -38.6729371, upper bound: 38.9460297
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 135.96
Output dim: 2, lower bound: -38.6300629, upper bound: 39.0040709

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 647

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0024557, upper bound: 38.5509837
time: 79.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8752920, upper bound: 38.6246631
time: 65.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 647

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6246631, upper bound: 38.8752920
time: 76.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.5509837, upper bound: 39.0024557
time: 78.92 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 158.33 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 158.33
Output dim: 2, lower bound: -39.0024557, upper bound: 38.5509837
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 158.33
Output dim: 2, lower bound: -38.8752920, upper bound: 38.6246631
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 158.33
Output dim: 2, lower bound: -38.6246631, upper bound: 38.8752920
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 158.33
Output dim: 2, lower bound: -38.5509837, upper bound: 39.0024557

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 632

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0011507, upper bound: 38.4645319
time: 76.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8952549, upper bound: 38.5483561
time: 77.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 632

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.4645319, upper bound: 38.8952549
time: 106.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.4645319, upper bound: 39.0011507
time: 84.63 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 193.37 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 193.37
Output dim: 2, lower bound: -39.0011507, upper bound: 38.4645319
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 193.37
Output dim: 2, lower bound: -38.8952549, upper bound: 38.5483561
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 193.37
Output dim: 2, lower bound: -38.4645319, upper bound: 38.8952549
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 193.37
Output dim: 2, lower bound: -38.4645319, upper bound: 39.0011507

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 729

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9325257, upper bound: 38.4633163
time: 76.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9999385, upper bound: 38.3957967
time: 82.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 729

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.3957967, upper bound: 38.9999385
time: 70.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.4633163, upper bound: 38.9325257
time: 74.20 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 147.33 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 147.33
Output dim: 2, lower bound: -38.9325257, upper bound: 38.4633163
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 147.33
Output dim: 2, lower bound: -38.9999385, upper bound: 38.3957967
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 147.33
Output dim: 2, lower bound: -38.3957967, upper bound: 38.9999385
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 147.33
Output dim: 2, lower bound: -38.4633163, upper bound: 38.9325257

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 649

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9984806, upper bound: 38.3274067
time: 72.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9052042, upper bound: 38.3916973
time: 74.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 649

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.3916973, upper bound: 38.9052042
time: 75.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.3274067, upper bound: 38.9984806
time: 73.94 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 151.49 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 151.49
Output dim: 2, lower bound: -38.9984806, upper bound: 38.3274067
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 151.49
Output dim: 2, lower bound: -38.9052042, upper bound: 38.3916973
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 151.49
Output dim: 2, lower bound: -38.3916973, upper bound: 38.9052042
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 151.49
Output dim: 2, lower bound: -38.3274067, upper bound: 38.9984806

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 727

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9329461, upper bound: 38.3263152
time: 83.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9973978, upper bound: 38.2609165
time: 83.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 727

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2609165, upper bound: 38.9973978
time: 77.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.3263152, upper bound: 38.9329461
time: 66.77 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 146.27 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 146.27
Output dim: 2, lower bound: -38.9329461, upper bound: 38.3263152
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 146.27
Output dim: 2, lower bound: -38.9973978, upper bound: 38.2609165
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 146.27
Output dim: 2, lower bound: -38.2609165, upper bound: 38.9973978
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 146.27
Output dim: 2, lower bound: -38.3263152, upper bound: 38.9329461

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 695

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9317520, upper bound: 38.2590234
time: 300.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9954137, upper bound: 38.1939192
time: 101.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 695

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1939192, upper bound: 38.9954137
time: 68.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.2590234, upper bound: 38.9317520
time: 128.74 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 199.57 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 199.57
Output dim: 2, lower bound: -38.9317520, upper bound: 38.2590234
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 199.57
Output dim: 2, lower bound: -38.9954137, upper bound: 38.1939192
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 199.57
Output dim: 2, lower bound: -38.1939192, upper bound: 38.9954137
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 199.57
Output dim: 2, lower bound: -38.2590234, upper bound: 38.9317520

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9637292, upper bound: 38.1913893
time: 108.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9946751, upper bound: 38.1751481
time: 76.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1751481, upper bound: 38.9946751
time: 91.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1913893, upper bound: 38.9637292
time: 82.30 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 176.74 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 176.74
Output dim: 2, lower bound: -38.9637292, upper bound: 38.1913893
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 176.74
Output dim: 2, lower bound: -38.9946751, upper bound: 38.1751481
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 12, time: 176.74
Output dim: 2, lower bound: -38.1751481, upper bound: 38.9946751
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 176.74
Output dim: 2, lower bound: -38.1913893, upper bound: 38.9637292

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2036
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 631

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9929148, upper bound: 38.1217459
time: 79.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9035296, upper bound: 38.1709231
time: 66.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2036
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 631

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1709231, upper bound: 38.9035296
time: 79.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1217459, upper bound: 38.9929148
time: 86.43 seconds

## Summary of splitting (split count: 12)
- Time for RS candidates: 168.09 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 13, time: 168.09
Output dim: 2, lower bound: -38.9929148, upper bound: 38.1217459
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 13, time: 168.09
Output dim: 2, lower bound: -38.9035296, upper bound: 38.1709231
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 13, time: 168.09
Output dim: 2, lower bound: -38.1709231, upper bound: 38.9035296
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 13, time: 168.09
Output dim: 2, lower bound: -38.1217459, upper bound: 38.9929148

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2035
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 633

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9921093, upper bound: 38.0801036
time: 80.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9384864, upper bound: 38.1200310
time: 89.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2035
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 633

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1200310, upper bound: 38.9384864
time: 145.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0801036, upper bound: 38.9921093
time: 71.93 seconds

## Summary of splitting (split count: 13)
- Time for RS candidates: 220.39 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 14, time: 220.39
Output dim: 2, lower bound: -38.9921093, upper bound: 38.0801036
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 14, time: 220.39
Output dim: 2, lower bound: -38.9384864, upper bound: 38.1200310
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 14, time: 220.39
Output dim: 2, lower bound: -38.1200310, upper bound: 38.9384864
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 14, time: 220.39
Output dim: 2, lower bound: -38.0801036, upper bound: 38.9921093

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2034
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1789

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9688860, upper bound: 38.0784349
time: 77.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9911722, upper bound: 38.0607329
time: 76.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2034
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1789

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0607329, upper bound: 38.9911722
time: 89.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0784349, upper bound: 38.9688860
time: 86.53 seconds

## Summary of splitting (split count: 14)
- Time for RS candidates: 178.78 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 15, time: 178.78
Output dim: 2, lower bound: -38.9688860, upper bound: 38.0784349
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 15, time: 178.78
Output dim: 2, lower bound: -38.9911722, upper bound: 38.0607329
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 15, time: 178.78
Output dim: 2, lower bound: -38.0607329, upper bound: 38.9911722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 15, time: 178.78
Output dim: 2, lower bound: -38.0784349, upper bound: 38.9688860

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2033
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 728

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9234795, upper bound: 38.0567332
time: 177.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9901586, upper bound: 38.0290795
time: 99.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2033
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 728

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0290795, upper bound: 38.9901586
time: 60.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0567332, upper bound: 38.9234795
time: 79.22 seconds

## Summary of splitting (split count: 15)
- Time for RS candidates: 142.44 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 16, time: 142.44
Output dim: 2, lower bound: -38.9234795, upper bound: 38.0567332
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 16, time: 142.44
Output dim: 2, lower bound: -38.9901586, upper bound: 38.0290795
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 16, time: 142.44
Output dim: 2, lower bound: -38.0290795, upper bound: 38.9901586
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 16, time: 142.44
Output dim: 2, lower bound: -38.0567332, upper bound: 38.9234795

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2032
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 735

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9894850, upper bound: 37.9959729
time: 74.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9568378, upper bound: 38.0284153
time: 70.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2032
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 735

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0284153, upper bound: 38.9568378
time: 93.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9959729, upper bound: 38.9894850
time: 69.41 seconds

## Summary of splitting (split count: 16)
- Time for RS candidates: 164.99 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 17, time: 164.99
Output dim: 2, lower bound: -38.9894850, upper bound: 37.9959729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 17, time: 164.99
Output dim: 2, lower bound: -38.9568378, upper bound: 38.0284153
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 17, time: 164.99
Output dim: 2, lower bound: -38.0284153, upper bound: 38.9568378
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 17, time: 164.99
Output dim: 2, lower bound: -37.9959729, upper bound: 38.9894850

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2031
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 697

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9234267, upper bound: 37.9945546
time: 86.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9884528, upper bound: 37.9283370
time: 81.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2031
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 697

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9283370, upper bound: 38.9884528
time: 77.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -37.9945546, upper bound: 38.9234267
time: 77.43 seconds

## Summary of splitting (split count: 17)
- Time for RS candidates: 156.89 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 18, time: 156.89
Output dim: 2, lower bound: -38.9234267, upper bound: 37.9945546
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 18, time: 156.89
Output dim: 2, lower bound: -38.9884528, upper bound: 37.9283370
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 18, time: 156.89
Output dim: 2, lower bound: -37.9283370, upper bound: 38.9884528
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 18, time: 156.89
Output dim: 2, lower bound: -37.9945546, upper bound: 38.9234267

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2030
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 597

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9564302, upper bound: 37.9275605
time: 66.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9876439, upper bound: 37.8968287
time: 69.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2030
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 597

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.8968287, upper bound: 38.9876439
time: 72.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -37.9275605, upper bound: 38.9564302
time: 77.04 seconds

## Summary of splitting (split count: 18)
- Time for RS candidates: 151.87 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 19, time: 151.87
Output dim: 2, lower bound: -38.9564302, upper bound: 37.9275605
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 19, time: 151.87
Output dim: 2, lower bound: -38.9876439, upper bound: 37.8968287
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 19, time: 151.87
Output dim: 2, lower bound: -37.8968287, upper bound: 38.9876439
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 19, time: 151.87
Output dim: 2, lower bound: -37.9275605, upper bound: 38.9564302

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2029
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1741

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9656627, upper bound: 37.8967703
time: 81.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9876415, upper bound: 37.8815161
time: 72.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2029
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1741

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.8815161, upper bound: 38.9876415
time: 68.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -37.8967703, upper bound: 38.9656627
time: 77.86 seconds

## Summary of splitting (split count: 19)
- Time for RS candidates: 148.51 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 20, time: 148.51
Output dim: 2, lower bound: -38.9656627, upper bound: 37.8967703
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 20, time: 148.51
Output dim: 2, lower bound: -38.9876415, upper bound: 37.8815161
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 20, time: 148.51
Output dim: 2, lower bound: -37.8815161, upper bound: 38.9876415
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 20, time: 148.51
Output dim: 2, lower bound: -37.8967703, upper bound: 38.9656627

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2028
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 713

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9358123, upper bound: 37.8768821
time: 62.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9862052, upper bound: 37.8518791
time: 74.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2028
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 713

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.8518792, upper bound: 38.9862052
time: 62.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -37.8768821, upper bound: 38.9358123
time: 76.04 seconds

## Summary of splitting (split count: 20)
- Time for RS candidates: 141.19 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 21, time: 141.19
Output dim: 2, lower bound: -38.9358123, upper bound: 37.8768821
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 21, time: 141.19
Output dim: 2, lower bound: -38.9862052, upper bound: 37.8518791
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 21, time: 141.19
Output dim: 2, lower bound: -37.8518792, upper bound: 38.9862052
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 21, time: 141.19
Output dim: 2, lower bound: -37.8768821, upper bound: 38.9358123

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2027
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 711

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9341241, upper bound: 37.8477804
time: 69.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9852772, upper bound: 37.8256305
time: 72.63 seconds

## Summary of splitting (split count: 21)
- Time for RS candidates: 144.95 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 22, time: 144.95
Output dim: 2, lower bound: -38.9341241, upper bound: 37.8477804
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 22, time: 144.95
Output dim: 2, lower bound: -38.9852772, upper bound: 37.8256305
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 21, time: 144.95
Output dim: 2, lower bound: -37.8518792, upper bound: 38.9862052

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 104.22 + 7114.96 = 7219.18 seconds
