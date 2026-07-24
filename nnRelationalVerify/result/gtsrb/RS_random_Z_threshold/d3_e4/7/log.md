## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 7)
Time budget: 7200 seconds
Split limit: 100
Threshold: 97.2066343617


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716)
1: (-70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341)
2: (-63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282)
3: (-72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957)
4: (-76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267)
5: (-68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710)
6: (-102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202)
7: (-84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920)
8: (-89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154)
9: (-78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873)
10: (-111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498)
11: (-111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485)
12: (-111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295)
13: (-110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117)
14: (-163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874)
15: (-92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756)
16: (-118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149)
17: (-164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765)
18: (-102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608)
19: (-85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756)
20: (-74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135)
21: (-104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721)
22: (-113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161)
23: (-86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248)
24: (-103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021)
25: (-91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324)
26: (-122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165)
27: (-104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583)
28: (-85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112)
29: (-119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958)
30: (-102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372)
31: (-106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931)
32: (-100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959)
33: (-141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360)
34: (-120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018)
35: (-120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321)
36: (-117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379)
37: (-164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464)
38: (-145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569)
39: (-168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181)
40: (-135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575)
41: (-100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632)
42: (-75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.89 + 116.16 = 119.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -97.3039383, upper bound: 97.3039383

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1734

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1707

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2985674, upper bound: 97.2828544
time: 254.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2828544, upper bound: 97.2985674
time: 126.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 381.09 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 381.09
Output dim: 5, lower bound: -97.2985674, upper bound: 97.2828544
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 381.09
Output dim: 5, lower bound: -97.2828544, upper bound: 97.2985674

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1585

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 645

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2926952, upper bound: 97.2715307
time: 142.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2715307, upper bound: 97.2769819
time: 107.57 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1713

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 617

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2800472, upper bound: 97.2957496
time: 138.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2800472, upper bound: 97.2972964
time: 134.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 274.51 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 274.51
Output dim: 5, lower bound: -97.2926952, upper bound: 97.2715307
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 274.51
Output dim: 5, lower bound: -97.2715307, upper bound: 97.2769819
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 274.51
Output dim: 5, lower bound: -97.2800472, upper bound: 97.2957496
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 274.51
Output dim: 5, lower bound: -97.2800472, upper bound: 97.2972964

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 689

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2718928, upper bound: 97.2504475
time: 103.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2716477, upper bound: 97.2506743
time: 1282.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1658

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2776156, upper bound: 97.2675224
time: 134.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2774109, upper bound: 97.2677069
time: 132.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1621

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 533

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2805277, upper bound: 97.2864033
time: 123.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2722287, upper bound: 97.2947119
time: 102.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1375

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 574

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2366275, upper bound: 97.2504663
time: 116.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2332446, upper bound: 97.2538535
time: 111.46 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 230.44 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 230.44
Output dim: 5, lower bound: -97.2718928, upper bound: 97.2504475
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 230.44
Output dim: 5, lower bound: -97.2716477, upper bound: 97.2506743
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 230.44
Output dim: 5, lower bound: -97.2776156, upper bound: 97.2675224
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 230.44
Output dim: 5, lower bound: -97.2774109, upper bound: 97.2677069
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 230.44
Output dim: 5, lower bound: -97.2805277, upper bound: 97.2864033
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 230.44
Output dim: 5, lower bound: -97.2722287, upper bound: 97.2947119
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 230.44
Output dim: 5, lower bound: -97.2366275, upper bound: 97.2504663
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 230.44
Output dim: 5, lower bound: -97.2332446, upper bound: 97.2538535

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 600

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1605

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2712758, upper bound: 97.2478003
time: 124.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2692736, upper bound: 97.2498218
time: 112.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1787

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2559974, upper bound: 97.2499141
time: 123.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2708838, upper bound: 97.2350003
time: 120.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 647

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 530

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2647017, upper bound: 97.2546971
time: 118.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2647017, upper bound: 97.2573174
time: 110.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1697

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1722

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2751944, upper bound: 97.2498042
time: 125.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2596554, upper bound: 97.2654479
time: 117.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 972

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 538

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2478758, upper bound: 97.2536907
time: 137.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2478758, upper bound: 97.2536907
time: 134.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 914

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1738

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2719019, upper bound: 97.2797606
time: 116.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2572583, upper bound: 97.2943852
time: 121.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 553

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2126795, upper bound: 97.2231466
time: 116.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2089925, upper bound: 97.2268391
time: 109.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 650

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 582

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2327070, upper bound: 97.2533298
time: 126.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2327142, upper bound: 97.2533230
time: 101.23 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 229.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 229.59
Output dim: 5, lower bound: -97.2712758, upper bound: 97.2478003
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 229.59
Output dim: 5, lower bound: -97.2692736, upper bound: 97.2498218
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 229.59
Output dim: 5, lower bound: -97.2559974, upper bound: 97.2499141
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 229.59
Output dim: 5, lower bound: -97.2708838, upper bound: 97.2350003
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 229.59
Output dim: 5, lower bound: -97.2647017, upper bound: 97.2546971
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 229.59
Output dim: 5, lower bound: -97.2647017, upper bound: 97.2573174
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 229.59
Output dim: 5, lower bound: -97.2751944, upper bound: 97.2498042
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 229.59
Output dim: 5, lower bound: -97.2596554, upper bound: 97.2654479
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 229.59
Output dim: 5, lower bound: -97.2478758, upper bound: 97.2536907
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 229.59
Output dim: 5, lower bound: -97.2478758, upper bound: 97.2536907
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 229.59
Output dim: 5, lower bound: -97.2719019, upper bound: 97.2797606
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 229.59
Output dim: 5, lower bound: -97.2572583, upper bound: 97.2943852
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 229.59
Output dim: 5, lower bound: -97.2126795, upper bound: 97.2231466
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 229.59
Output dim: 5, lower bound: -97.2089925, upper bound: 97.2268391
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 229.59
Output dim: 5, lower bound: -97.2327070, upper bound: 97.2533298
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 229.59
Output dim: 5, lower bound: -97.2327142, upper bound: 97.2533230

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1633

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 763

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2707287, upper bound: 97.2397277
time: 133.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2631823, upper bound: 97.2472499
time: 105.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 842

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1675

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2656350, upper bound: 97.2396169
time: 132.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2591482, upper bound: 97.2461797
time: 113.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 673

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2436073, upper bound: 97.2482142
time: 170.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2542811, upper bound: 97.2376071
time: 113.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 586

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1626

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2669452, upper bound: 97.2279328
time: 138.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2637578, upper bound: 97.2311337
time: 114.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 824

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 868

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2338496, upper bound: 97.2546899
time: 129.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2673017, upper bound: 97.2389063
time: 110.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 687

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 517

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2639775, upper bound: 97.2539064
time: 98.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2613093, upper bound: 97.2565803
time: 116.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 613

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 566

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2748607, upper bound: 97.2382518
time: 136.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2636263, upper bound: 97.2494704
time: 113.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1740

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 672

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2551492, upper bound: 97.2645411
time: 139.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2587604, upper bound: 97.2609530
time: 142.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716
1: -70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341
2: -63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282
3: -72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957
4: -76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267
5: -68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710
6: -102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202
7: -84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920
8: -89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154
9: -78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873
10: -111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498
11: -111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485
12: -111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295
13: -110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117
14: -163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874
15: -92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756
16: -118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149
17: -164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765
18: -102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608
19: -85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756
20: -74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135
21: -104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721
22: -113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161
23: -86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248
24: -103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021
25: -91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324
26: -122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165
27: -104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583
28: -85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112
29: -119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958
30: -102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372
31: -106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931
32: -100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959
33: -141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360
34: -120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018
35: -120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321
36: -117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379
37: -164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464
38: -145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569
39: -168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181
40: -135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575
41: -100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632
42: -75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1311
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 524

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1784

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2475638, upper bound: 97.2413382
time: 114.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2355781, upper bound: 97.2533770
time: 151.84 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 268.85 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2707287, upper bound: 97.2397277
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2631823, upper bound: 97.2472499
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2656350, upper bound: 97.2396169
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2591482, upper bound: 97.2461797
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2436073, upper bound: 97.2482142
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2542811, upper bound: 97.2376071
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2669452, upper bound: 97.2279328
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2637578, upper bound: 97.2311337
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2338496, upper bound: 97.2546899
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2673017, upper bound: 97.2389063
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2639775, upper bound: 97.2539064
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2613093, upper bound: 97.2565803
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2748607, upper bound: 97.2382518
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2636263, upper bound: 97.2494704
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2551492, upper bound: 97.2645411
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2587604, upper bound: 97.2609530
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2475638, upper bound: 97.2413382
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 268.85
Output dim: 5, lower bound: -97.2355781, upper bound: 97.2533770
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 268.85
Output dim: 5, lower bound: -97.2478758, upper bound: 97.2536907
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 268.85
Output dim: 5, lower bound: -97.2719019, upper bound: 97.2797606
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 268.85
Output dim: 5, lower bound: -97.2572583, upper bound: 97.2943852
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 268.85
Output dim: 5, lower bound: -97.2126795, upper bound: 97.2231466
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 268.85
Output dim: 5, lower bound: -97.2089925, upper bound: 97.2268391
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 268.85
Output dim: 5, lower bound: -97.2327070, upper bound: 97.2533298
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 268.85
Output dim: 5, lower bound: -97.2327142, upper bound: 97.2533230

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 119.05 + 7253.94 = 7372.98 seconds
