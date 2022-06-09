import os
'''
Statistical number of images per classes
'''
path = '/home/fitmta/Binh53/DoAnTotNghiep/dataset/temp/Core_DataProcessing/4_Png/Train'
def visualize(path):
    for item in os.listdir(path):
        subItemPath = path + '/' + item
        count = 0
        for img in os.listdir(subItemPath):
            count += 1
            if count >= 500:
                os.remove(os.path.join(subItemPath,img))
        print(f'Number of images in class "{item}" ===> {count}')
def removeEmptyFolder(filepath):
    for item in os.listdir(filepath):
        pathItem = os.path.join(filepath, item)
        if not os.listdir(pathItem):
            os.rmdir(pathItem)
# removeEmptyFolder('/home/fitmta/Binh53/DoAnTotNghiep/dataset/temp/Core_DataProcessing/3_ProcessedSession/FilteredSession/Train')
# removeEmptyFolder('/home/fitmta/Binh53/DoAnTotNghiep/dataset/temp/Core_DataProcessing/3_ProcessedSession/FilteredSession/Test')
visualize(path)