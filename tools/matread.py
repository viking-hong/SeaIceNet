import h5py

# 打开.mat文件
with h5py.File('D:/SI-STSAR-7/20210123_6s.mat', 'r') as file:
    # 查看文件中的数据集名称

    your_data = file['data'][:]
    print(your_data)