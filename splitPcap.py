from scapy.all import rdpcap
from scapy.error import Scapy_Exception
import os
import re 
import errno 
import shutil
PCAP_DIR = '/home/fitmta/Binh53/DoAnTotNghiep/dataset/temp/Core_DataProcessing/2_Session/AllLayers_Pkts'
List_of_Deivce = '/home/fitmta/Binh53/DoAnTotNghiep/dataset/temp/filelist.txt'
path_device = '2_Session/AllLayers_MAC' # or 2_Session/AllLayers_MAC_L7
def mkdir_p(list_device):
    for mac, device in list_device.items():
        dir_full = os.path.join(path_device, device)
        try:
            if os.path.exists(dir_full) == False:
                os.makedirs(dir_full)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(dir_full):
                pass
            else:
                raise
def extractDevice(fileDevice):
    dict_mac = {}
    with open(fileDevice) as file:
        text = file.read()
        result = text.strip().split('\n')
        for row in result[2:]: # remove title so we start from index 2
            res = row.split('\t') # Name - MAC - connection type
            device_name = (res[0] + ('_' + res[2] if len(res) == 3 else '')).strip()
            mac_addr = res[1].strip()
            dict_mac[mac_addr] = device_name
    return dict_mac
def extractMACperDevice(fileDevice):
    dict_mac = {}
    with open(fileDevice) as file:
        text = file.read()
        result = text.strip().split('\n')
        for row in result[2:]: # remove title so we start from index 2
            res = row.split('\t') # Name - MAC - connection type
            device_name = (res[0] + ('_' + res[2] if len(res) == 3 else '')).strip()
            mac_addr = res[1].strip().replace(':','').upper()
            dict_mac[mac_addr] = device_name
    return dict_mac
def loadPcap(pcap):
    if os.path.isfile(pcap):
        try:
            packet = rdpcap(pcap)
            src_mac = None
            for frame in packet: # packet is 1
                src_mac = frame['Ether']
                break
            return src_mac.src
        except Scapy_Exception as msg:
            print(str(msg))
    else:
        print('You sure that \'s the right file location???')  
def splitByMAC(pcap_dir, dict_mac):
    for item in os.listdir(pcap_dir):
        dir_item = os.path.join(pcap_dir, item)
        try:
            src_MAC = loadPcap(dir_item)
            if src_MAC in dict_mac:
                dir_full = os.path.join(path_device, dict_mac[src_MAC])
                shutil.copy(dir_item, dir_full)
        except:
            continue
    print('------ Done split by MAC address ------')
def statisticFile(pcap_dir):
    print('------ Number of files in folder ------')
    sum = 0
    for item in os.listdir(pcap_dir):
        dir_item = os.path.join(pcap_dir, item)
        sumDir = len(os.listdir(dir_item))
        if sumDir == 0:
            os.rmdir(dir_item)
        else:
            sum += sumDir
            print(item, sumDir)
    print(f'Total file: {sum} files')

def removeHeader(pcap):
    if os.path.isfile(pcap):
        try:
            packet = rdpcap(pcap)
            for frame in packet:
                e = frame['Raw']
                # print(e)
                # break
        except Scapy_Exception as msg:
            print(str(msg))
    else:
        print('You sure that \'s the right file location???')  
def splitMacAllLayer():
    dict_macDevice = extractDevice(List_of_Deivce)
    mkdir_p(dict_macDevice)
    splitByMAC(PCAP_DIR, dict_macDevice)
    statisticFile(path_device)
def splitFolderMacOnlyM7():
    global path_device 
    path_device = '2_Session/AllLayers_MAC_L7'
    dict_macDevice = extractDevice(List_of_Deivce)
    mkdir_p(dict_macDevice)
def splitMacOnlyM7():
    global path_device 
    path_device = '2_Session/AllLayers_MAC_L7'
    dict_macDevice = extractMACperDevice(List_of_Deivce)
    print(dict_macDevice)
    for mac, device in dict_macDevice.items():
        dir_full = os.path.join(path_device, device)
        if os.path.exists(dir_full):
            for item in os.listdir(dir_full):     
                #filename : iot.pcap.TCP_176-32-98-204_443_192-168-1-240_34635_00171_20160923135321.pcap.MAC_14CC205133EA
                mac_addr = item.strip().split('_')[-1].split('.')[0] # get MAC address
                if mac_addr != mac:
                    myfile = os.path.join(dir_full, item)
                    if os.path.isfile(myfile):
                        os.remove(myfile)
                    else:    ## Show an error ##
                        print("Error: %s file not found" % myfile)


# print(removeHeader('/home/fitmta/Binh53/DoAnTotNghiep/dataset/fixed_mac.pcap'))
# print(len(os.listdir('/home/fitmta/Binh53/DoAnTotNghiep/dataset/temp/Core_DataProcessing/2_Session/AllLayers_SplitMac/')))
# print(len(os.listdir('/home/fitmta/Binh53/DoAnTotNghiep/dataset/temp/Core_DataProcessing/2_Session/AllLayers_Pkts/fixed_dumpfile')))

'''
Applied for split All Layer
'''
# splitMacAllLayer() 
'''
Applied for create folder L7
'''
# splitFolderMacOnlyM7() 
'''
Applied for split L7
'''
splitMacOnlyM7() 