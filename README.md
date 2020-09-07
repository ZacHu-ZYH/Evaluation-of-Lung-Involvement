# Evaluation of Lung Involvement in COVID-19 based on SE_ResNeXt
Evaluation of Lung Involvement in COVID-19 Pneumonia Based on Ultrasound Images

## Environment
* Tensorflow 1.x
* Python 3.x
* tflearn (If you are easy to use ***global average pooling***, you should install ***tflearn***)
## Paper link
https://doi.org/10.21203/rs.3.rs-70092/v1
## Pretrained models
link：https://pan.baidu.com/s/1uercarSo7uQehmf8S6nHKQ 
code：nhqq
## Data link
multi-center data will be released after the paper was accepted
## Data preparing

* First, we obtained K-Means and Gradient modalities by running

```
python K-means_Grad.py
```

* Then, We preprocessed the dataset by running

```
python makecifar10_mix.py
```

* Training

```
python SE_ResNeXt.py
```
* Testing

```
python SE_ResNeXt_test.py
```
