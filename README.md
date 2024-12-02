# PyITS

<a href="https://github.com/Master-PLC/PyITS">
    <img src="./docs/logo1.svg" width="1000" align="">
</a>

<h3 align="center">Welcome to PyITS</h3>

<p align="center"><i>A Unified Python Toolkit for Industrial Time-Series Analytics</i></p>

<p align="center">
    <a href="https://github.com/Master-PLC/PyITS">
       <img alt="Python version" src="https://img.shields.io/badge/Python-v3.8+-E97040?logo=python&logoColor=white">
    </a>
    <a href="https://github.com/Master-PLC/PyITS">
        <img alt="powered by Pytorch" src="https://img.shields.io/badge/PyTorch-v2.1+-E97040?logo=pytorch&logoColor=white">
    </a>
    <!-- <a href="https://github.com/Master-PLC/PyITS">
        <img alt="the latest release version" src="https://img.shields.io/github/v/release/Master-PLC/PyITS?color=EE781F&include_prereleases&label=Release&logo=github&logoColor=white">
    </a> -->
    <a href="https://github.com/Master-PLC/PyITS">
        <img alt="MIT license" src="https://img.shields.io/badge/License-MIT-E9BB41?logo=opensourceinitiative&logoColor=white">
    </a>
    <a href="https://github.com/Master-PLC/PyITS">
        <img alt="Community" src="https://img.shields.io/badge/join_us-community!-C8A062">
    </a>
    <a href="https://github.com/Master-PLC/PyITS">
        <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/Master-PLC/PyITS?color=D8E699&label=Contributors&logo=GitHub">
    </a>
    <a href="https://star-history.com/#Master-PLC/PyITS">
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Master-PLC/PyITS?logo=None&color=6BB392&label=%E2%98%85%20Stars">
    </a>
    <a href="https://github.com/Master-PLC/PyITS/network/members">
        <img alt="GitHub Repo forks" src="https://img.shields.io/github/forks/Master-PLC/PyITS?logo=forgejo&logoColor=black&label=Forks">
    </a>
    <!-- <a href="https://codeclimate.com/github/Master-PLC/PyITS"> -->
        <!-- <img alt="Code Climate maintainability" src="https://img.shields.io/codeclimate/maintainability-percentage/Master-PLC/PyITS?color=3C7699&label=Maintainability&logo=codeclimate"> -->
    </a>
    <!-- <a href="https://coveralls.io/github/Master-PLC/PyITS"> -->
        <!-- <img alt="Coveralls coverage" src="https://img.shields.io/coverallsCoverage/github/Master-PLC/PyITS?branch=main&logo=coveralls&color=75C1C4&label=Coverage"> -->
    </a>
    <!-- <a href="https://github.com/Master-PLC/PyITS/actions/workflows/testing_ci.yml"> -->
        <!-- <img alt="GitHub Testing" src="https://img.shields.io/github/actions/workflow/status/Master-PLC/PyITS/testing_ci.yml?logo=circleci&color=C8D8E1&label=CI"> -->
    </a>
    <!-- <a href="https://docs.pypots.com">
        <img alt="Docs building" src="https://img.shields.io/readthedocs/pypots?logo=readthedocs&label=Docs&logoColor=white&color=395260">
    </a>
    <a href="https://anaconda.org/conda-forge/pypots">
        <img alt="Conda downloads" src="https://img.shields.io/endpoint?url=https://pypots.com/figs/downloads_badges/conda_pypots_downloads.json">
    </a>
    <a href="https://pepy.tech/project/pypots">
        <img alt="PyPI downloads" src="https://img.shields.io/endpoint?url=https://pypots.com/figs/downloads_badges/pypi_pypots_downloads.json">
    </a>
    <a href="https://arxiv.org/abs/2305.18811">
        <img alt="arXiv DOI" src="https://img.shields.io/badge/DOI-10.48550/arXiv.2305.18811-F8F7F0">
    </a>
    <a href="https://github.com/Master-PLC/PyITS/blob/main/README_zh.md">
        <img alt="README in Chinese" src="https://pypots.com/figs/pypots_logos/readme/CN.svg">
    </a> -->
   <a href="https://github.com/Master-PLC/PyITS/blob/main/README.md">
        <img alt="README in English" src="https://pypots.com/figs/pypots_logos/readme/US.svg">
    </a>
    <!-- <a href="https://github.com/Master-PLC/PyITS">
        <img alt="PyPOTS Hits" src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPyPOTS%2FPyPOTS&count_bg=%23009A0A&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits&edge_flat=false">
    </a> -->
</p>

⦿ `Motivation`: Industrial Time-Series (ITS) analytics is essential across diverse sectors, including manufacturing, energy, transportation, and healthcare. While general time-series models have made significant advancements, ITS encompasses specialized tasks such as fault diagnosis, process monitoring, soft sensing, RUC prediction, and predictive maintenance. These tasks require unique data processing, training, and evaluation protocols, making existing toolkits unsuitable for ITS applications. Furthermore, implementation details for these protocols are often not disclosed in current literature. Therefore, the area of ITS analytics needs a dedicated toolkit. PyITS is created to fill in this blank.


⦿ `Mission`: PyITS is designed to be a versatile toolbox that streamlines the application of modern machine learning techniques for Industrial Time-Series (ITS) analytics. Our mission is twofold: (1) Enable engineers to quickly implement and compare existing algorithms, facilitating the efficient completion of ITS projects. (2) Provide researchers with a robust collection of updated baselines and standardized experimental protocols, allowing them to focus on algorithmic innovation. PyITS continuously integrates both classical and state-of-the-art machine learning algorithms, offering unified APIs across various models and tasks. Additionally, we provide comprehensive documentation and interactive tutorials to guide beginners through different applications.

🤗 **Please** star this repo to help others notice PyPOTS if you think it is a useful toolkit.
This really means a lot to our open-source research. Thank you!

## ❖ Available Algorithms

PyITS supports fault diagnosis, process monitoring, soft sensing, RUC prediction, and predictive maintenance tasks on ITS datasets. 
The table below shows the availability of each algorithm for different tasks. 
The symbol `✅` indicates the algorithm is available for the corresponding task (note that some models are specifically designed for typical tasks, and PyITS modifies the data processing and output protocol to extend the spectrum of supported tasks).

Suppose T is the historical length, at the time-step $t$, the supported tasks are discribed as follows:
**`SS`**: Soft sensor, which aims to estimate the value of the quality variable ($y_t$). Available input data include the historical values of process variables $x_{t-\mathrm{T}+1:t}$ and quality variable $y_{t-\mathrm{T}+1:t-1}$.
**`PM`**: Process monitoring, which aims to predicts the next-step quality variable ($y_{t+1}$). Available input data include the historical values of process variables $x_{t-\mathrm{T}+1:t}$ and quality variable $y_{t-\mathrm{T}+1:t}$.
**`FD`**: Fault diagnosis, which aims to determines the correct status of a given patch, including normal and various fault types. Available input data include the historical values of all variables $z_{t-\mathrm{T}+1:t}$.
**`RUL`**: RUL prediction, which aims to forecasts the remaining time before a fault occurs. Available input data include the historical values of all variables $z_{t-\mathrm{T}+1:t}$.
**`PM`**: Predictive maintainance, which aims to estimates whether a fault occurrs within a specified future window. Available input data include the historical values of all variables $z_{t-\mathrm{T}+1:t}$.

All paper references and relevant links are provided at the end of this document.

| Type 	| Model 	| Paper 	| SS 	| PM 	| FD 	| RUL 	| PD 	| Remarks 	|
|---	|---	|---	|---	|---	|---	|---	|---	|---	|
| Foundation model 	| Autoformer 	| NIPS2021 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Transformer-based time-series foundational model 	|
| Foundation model 	| Crossformer 	| ICLR2023 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Transformer-based time-series foundational model 	|
| Foundation model 	| ETSformer 	| arxiv 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Transformer-based time-series foundational model 	|
| Foundation model 	| FEDformer 	| ICML2022 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Transformer-based time-series foundational model 	|
| Foundation model 	| Informer 	| AAAI2021 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Transformer-based time-series foundational model 	|
| Foundation model 	| Nonstationary Transformer 	| NIPS2022 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Transformer-based time-series foundational model 	|
| Foundation model 	| PatchTST 	| ICLR2023 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Transformer-based time-series foundational model 	|
| Foundation model 	| Pathformer 	| ICLR2024 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Transformer-based time-series foundational model 	|
| Foundation model 	| PAttn 	| NIPS2024 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Transformer-based time-series foundational model 	|
| Foundation model 	| Pyraformer 	| ICLR2022 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Transformer-based time-series foundational model 	|
| Foundation model 	| Reformer 	| ICLR2020 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Transformer-based time-series foundational model 	|
| Foundation model 	| TimeXer 	| NIPS2024 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Transformer-based time-series foundational model 	|
| Foundation model 	| Transformer 	| NIPS2017 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Transformer-based time-series foundational model 	|
| Foundation model 	| iTransformer 	| ICLR2024 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Transformer-based time-series foundational model 	|
| Foundation model 	| FITS 	| arxiv 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| MLP-based time-series foundational model 	|
| Foundation model 	| DLinear 	| AAAI2023 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| MLP-based time-series foundational model 	|
| Foundation model 	| LightTS 	| PACM2023 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| MLP-based time-series foundational model 	|
| Foundation model 	| FreTS 	| NIPS2023 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| MLP-based time-series foundational model 	|
| Foundation model 	| TiDE 	| arxiv 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| MLP-based time-series foundational model 	|
| Foundation model 	| TimeMixer 	| ICLR2024 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| MLP-based time-series foundational model 	|
| Foundation model 	| Triformer 	| IJCAI2022 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| MLP-based time-series foundational model 	|
| Foundation model 	| TSMixer 	| TMLR2023 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| MLP-based time-series foundational model 	|
| Foundation model 	| Mamba 	| arxiv 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| RNN-based time-series foundational model 	|
| Foundation model 	| LSTM 	|  	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| RNN-based time-series foundational model 	|
| Foundation model 	| SegRNN 	| arxiv 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| RNN-based time-series foundational model 	|
| Foundation model 	| MICN 	| ICLR2023 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| CNN-based time-series foundational model 	|
| Foundation model 	| SCINet 	| NIPS2022 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| CNN-based time-series foundational model 	|
| Foundation model 	| TCN 	| arxiv 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| CNN-based time-series foundational model 	|
| Foundation model 	| TimesNet 	| ICLR2023 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| CNN-based time-series foundational model 	|
| Foundation model 	| Koopa 	| NIPS2023 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Other time-series foundational model 	|
| Foundation model 	| FiLM 	| NIPS2022 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Other time-series foundational model 	|
| Task-specific model 	| DLSTM 	| TII2022 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Originally developed for SS task 	|
| Task-specific model 	| DTGRU 	| TII2023 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Originally developed for RUL task 	|
| Task-specific model 	| AdaNet 	| TII2024 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Originally developed for RUL task 	|
| Task-specific model 	| DeepPLS 	| TNNLS2023 	| ✅ 	| ✅ 	|  	| ✅ 	|  	| Originally developed for PM task 	|
| Task-specific model 	| RSN 	| TII2023 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Originally developed for FD task 	|
| Task-specific model 	| MCN 	| TSMC2024 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Originally developed for FD task 	|
| Task-specific model 	| MCTAN 	| TNNLS2023 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Originally developed for SS task 	|
| Task-specific model 	| DLformer 	| TNNLS2024 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Originally developed for RUL task 	|
| Task-specific model 	| TR-LT 	| TII2022 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Originally developed for RUL task 	|

💯 Contribute your model right now to increase your research impact! Your work will be widely used and cited by the community.
Refer to the `models/Transformer.py` and `estimator/foundation/soft_sensor_estimator.py` to see how to include your model and estimator in PyITS.

## ❖ PyITS Composition

At PyPOTS, things are related to coffee, which we're familiar with. Yes, this is a coffee universe!
As you can see, there is a coffee pot in the PyPOTS logo. And what else? Please read on ;-)



👈 Time series datasets are taken as coffee beans at PyPOTS, and POTS datasets are incomplete coffee beans with missing
parts that have their own meanings. To make various public time-series datasets readily available to users,
<i>Time Series Data Beans (TSDB)</i> is created to make loading time-series datasets super easy!
Visit [TSDB](https://github.com/WenjieDu/TSDB) right now to know more about this handy tool 🛠, and it now supports a
total of 172 open-source datasets!



👉 To simulate the real-world data beans with missingness, the ecosystem library
[PyGrinder](https://github.com/WenjieDu/PyGrinder), a toolkit helping grind your coffee beans into incomplete ones, is
created. Missing patterns fall into three categories according to Robin's theory[^13]:
MCAR (missing completely at random), MAR (missing at random), and MNAR (missing not at random).
PyGrinder supports all of them and additional functionalities related to missingness.
With PyGrinder, you can introduce synthetic missing values into your datasets with a single line of code.



👈 To fairly evaluate the performance of PyPOTS algorithms, the benchmarking suite
[BenchPOTS](https://github.com/WenjieDu/BenchPOTS) is created, which provides standard and unified data-preprocessing
pipelines to prepare datasets for measuring the performance of different POTS algorithms on various tasks.



👉 Now the beans, grinder, and pot are ready, please have a seat on the bench and let's think about how to brew us a cup
of coffee. Tutorials are necessary! Considering the future workload, PyPOTS tutorials are released in a single repo,
and you can find them in [BrewPOTS](https://github.com/WenjieDu/BrewPOTS).
Take a look at it now, and learn how to brew your POTS datasets!

<p align="center">
<a href="https://pypots.com/ecosystem/">
    <img src="https://pypots.com/figs/pypots_logos/Ecosystem/PyPOTS_Ecosystem_Pipeline.png" width="95%"/>
</a>
<br>
<b> ☕️ Welcome to the universe of PyPOTS. Enjoy it and have fun!</b>
</p>

## ❖ Installation

We will release a package on PyPI and anaconda very soon. Before that, you can install PyITS by cloning it.
``` bash
git clone https://github.com/Master-PLC/PyITS.git
```

## ❖ Usage

Besides [BrewPOTS](https://github.com/WenjieDu/BrewPOTS), you can also find a simple and quick-start tutorial notebook
on Google Colab
<a href="https://colab.research.google.com/drive/1HEFjylEy05-r47jRy0H9jiS_WhD0UWmQ">
<img src="https://img.shields.io/badge/GoogleColab-PyPOTS_Tutorials-F9AB00?logo=googlecolab&logoColor=white" alt="Colab tutorials" align="center"/>
</a>. If you have further questions, please refer to PyPOTS documentation [docs.pypots.com](https://docs.pypots.com).
You can also [raise an issue](https://github.com/Master-PLC/PyITS/issues) or [ask in our community](#-community).

We present you a usage example of imputing missing values in time series with PyPOTS below, you can click it to view.

<details open>
<summary><b>Click here to see an example applying SAITS on PhysioNet2012 for imputation:</b></summary>

``` python
# Data preprocessing. Tedious, but PyPOTS can help.
import numpy as np
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar
from pypots.data import load_specific_dataset
data = load_specific_dataset('physionet_2012')  # PyPOTS will automatically download and extract it.
X = data['X']
num_samples = len(X['RecordID'].unique())
X = X.drop(['RecordID', 'Time'], axis = 1)
X = StandardScaler().fit_transform(X.to_numpy())
X = X.reshape(num_samples, 48, -1)
X_ori = X  # keep X_ori for validation
X = mcar(X, 0.1)  # randomly hold out 10% observed values as ground truth
dataset = {"X": X}  # X for model input
print(X.shape)  # (11988, 48, 37), 11988 samples and each sample has 48 time steps, 37 features

# Model training. This is PyPOTS showtime.
from pypots.imputation import SAITS
from pypots.utils.metrics import calc_mae
saits = SAITS(n_steps=48, n_features=37, n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=128, dropout=0.1, epochs=10)
# Here I use the whole dataset as the training set because ground truth is not visible to the model, you can also split it into train/val/test sets
saits.fit(dataset)  # train the model on the dataset
imputation = saits.impute(dataset)  # impute the originally-missing values and artificially-missing values
indicating_mask = np.isnan(X) ^ np.isnan(X_ori)  # indicating mask for imputation error calculation
mae = calc_mae(imputation, np.nan_to_num(X_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
saits.save("save_it_here/saits_physionet2012.pypots")  # save the model for future use
saits.load("save_it_here/saits_physionet2012.pypots")  # reload the serialized model file for following imputation or training
```

</details>

## ❖ Contribution

You're very welcome to contribute to this exciting project!

By committing your code, you'll

1. make your well-established model out-of-the-box for PyPOTS users to run, and help your work obtain more exposure and impact.
2. become one of PyITS contributors and be listed as a volunteer developer;
3. get mentioned in PyPOTS [release notes](https://github.com/Master-PLC/PyITS/releases);

Refer to the `models/Transformer.py` and `estimator/foundation/soft_sensor_estimator.py` to see how to include your model and estimator in PyITS.


## ❖ Community

We care about the feedback from our users, so we're building PyPOTS community on

- [Slack](https://join.slack.com/t/pypots-org/shared_invite/zt-1gq6ufwsi-p0OZdW~e9UW_IA4_f1OfxA). General discussion,
  Q&A, and our development team are here;
- [LinkedIn](https://www.linkedin.com/company/pypots). Official announcements and news are here;
- [WeChat (微信公众号)](https://mp.weixin.qq.com/s/X3ukIgL1QpNH8ZEXq1YifA). We also run a group chat on WeChat,
  and you can get the QR code from the official account after following it;

If you have any suggestions or want to contribute ideas or share time-series related papers, join us and tell.
PyPOTS community is open, transparent, and surely friendly. Let's work together to build and improve PyPOTS!
