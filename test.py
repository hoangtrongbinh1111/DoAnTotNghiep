from scapy.all import rdpcap
from scapy.error import Scapy_Exception
import os
import re 
import errno 
import shutil

capture = rdpcap('/home/fitmta/Binh53/DoAnTotNghiep/dataset/test.pcap')
for packet in capture:
    print(packet)
    break
