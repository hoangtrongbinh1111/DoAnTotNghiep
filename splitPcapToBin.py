import os
import time
from scapy.all import *
import numpy as np
import os
import time
# this is 1 session TCP
path_device = '2_Session/AllLayers_MAC' # or 2_Session/AllLayers_MAC_L7
savepath_device = '2_Session/test'
for device in os.listdir(path_device):
    savepath_deviceItem = os.path.join(savepath_device, device)
    if (os.path.exists(savepath_deviceItem) == False):
        os.mkdir(savepath_deviceItem)
    listdir_device = os.path.join(path_device, device)
    counter = 0
    ts = time.time()
    for item in os.listdir(listdir_device):
        listdir_item = os.path.join(listdir_device, item)
        pkts = rdpcap(listdir_item)
        for p in pkts:
            content = bytes(p['TCP'].payload)
            if content != b'':
                with open(f"{savepath_deviceItem}/iotTrace_{counter}.bin", "wb") as binary_file:
                    binary_file.write(content)
                counter = counter + 1
    dt = time.time() - ts
    print(f'Done in {listdir_device} with {dt} sec !!!')
# pkts = rdpcap(filename)
# count = 0
# for p in pkts:
#     if count == 3:
#         with open("my_file.bin", "wb") as binary_file:
#             # Write bytes to file
#             binary_file.write(bytes(p['TCP'].payload))
#         break
#     count = count + 1