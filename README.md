# RDN

Project tăng cường độ phân giải ảnh . Paper["Residual Dense Network for Image Super-Resolution"](https://arxiv.org/abs/1802.08797).

## Requirements

- PyTorch 1.0.0
- Numpy 1.15.4
- Pillow 5.4.1
- h5py 2.8.0


## Train
Vào file notebook SuperResolution.ipynb

Load checkpoint tại đây 

Tạo và cd vào folder Dataset
Uncomment phần wget và unzip để lấy dữ liệu đã được chia thành các folder HR và LR tương ứng

Tinh chỉnh các siêu tham số 
-- num-featue : số feature ở tầng Conv đầu tiên
-- growth-rate : số feature trong các DenseBlock
-- scale : hệ số phóng
-- num-blocks : số block RDB trong 1 layer 
-- num-layers : số layer
-- lr : learning rate
-- patch_size : kích thước ảnh được cắt nhỏ
-- batch_size 
-- epochs 



