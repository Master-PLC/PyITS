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
   <a href="https://github.com/Master-PLC/PyITS/blob/main/README.md">
        <img alt="README in English" src="https://pypots.com/figs/pypots_logos/readme/US.svg">
    </a>
</p>

⦿ `Motivation`: Industrial Time-Series (ITS) analytics is essential across diverse sectors, including manufacturing, energy, transportation, and healthcare. While general time-series models have made significant advancements, ITS encompasses specialized tasks such as fault diagnosis, process monitoring, soft sensing, RUC prediction, and predictive maintenance. These tasks require unique data processing, training, and evaluation protocols, making existing toolkits unsuitable for ITS applications. Furthermore, implementation details for these protocols are often not disclosed in current literature. Therefore, the area of ITS analytics needs a dedicated toolkit. PyITS is created to fill in this blank.


⦿ `Mission`: PyITS is designed to be a versatile toolbox that streamlines the application of modern machine learning techniques for Industrial Time-Series (ITS) analytics. Our mission is twofold: (1) Enable engineers to quickly implement and compare existing algorithms, facilitating the efficient completion of ITS projects. (2) Provide researchers with a robust collection of updated baselines and standardized experimental protocols, allowing them to focus on algorithmic innovation. PyITS continuously integrates both classical and state-of-the-art machine learning algorithms, offering unified APIs across various models and tasks. Additionally, we provide comprehensive documentation and interactive tutorials to guide beginners through different applications.

🤗 **Please** star this repo to help others notice PyITS if you think it is a useful toolkit.
This really means a lot to our open-source research. Thank you!

## ❖ Available Algorithms

PyITS supports fault diagnosis, process monitoring, soft sensing, RUC prediction, and predictive maintenance tasks on ITS datasets. Suppose T is the historical length, at the time-step $t$, the supported tasks are discribed as follows:
- **`SS`**: Soft sensor, which aims to estimate the value of the quality variable ($y_t$). Available input data include the historical values of process variables $x_{t-\mathrm{T}+1:t}$ and quality variable $y_{t-\mathrm{T}+1:t-1}$.
- **`PM`**: Process monitoring, which aims to predicts the next-step quality variable ($y_{t+1}$). Available input data include the historical values of process variables $x_{t-\mathrm{T}+1:t}$ and quality variable $y_{t-\mathrm{T}+1:t}$.
- **`FD`**: Fault diagnosis, which aims to determines the correct status of a given patch, including normal and various fault types. Available input data include the historical values of all variables $z_{t-\mathrm{T}+1:t}$.
- **`RUL`**: RUL prediction, which aims to forecasts the remaining time before a fault occurs. Available input data include the historical values of all variables $z_{t-\mathrm{T}+1:t}$.
- **`PdM`**: Predictive maintainance, which aims to estimates whether a fault occurrs within a specified future window. Available input data include the historical values of all variables $z_{t-\mathrm{T}+1:t}$.

The table below shows the availability of each algorithm for different tasks. 
The symbol `✅` indicates the algorithm is available for the corresponding task (note that some models are specifically designed for typical tasks, and PyITS modifies the data processing and output protocol to extend the spectrum of supported tasks).



| Type 	| Model 	| Paper 	| SS 	| PM 	| FD 	| RUL 	| PdM 	| Remarks 	|
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
| Task-specific model 	| DeepPLS 	| TNNLS2023 	| ✅ 	| ✅ 	|  	| ✅ 	|  	| Originally developed for PdM task 	|
| Task-specific model 	| RSN 	| TII2023 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Originally developed for FD task 	|
| Task-specific model 	| MCN 	| TSMC2024 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Originally developed for FD task 	|
| Task-specific model 	| MCTAN 	| TNNLS2023 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Originally developed for SS task 	|
| Task-specific model 	| DLformer 	| TNNLS2024 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Originally developed for RUL task 	|
| Task-specific model 	| TR-LT 	| TII2022 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| ✅ 	| Originally developed for RUL task 	|

✨ Contribute your model right now to enhance your research impact! Your work will be widely used and cited by the community.
Refer to the `models/Transformer.py` to see how to devise your model with PyITS.

## ❖ PyITS Composition

The PyITS framework consists of three main components: data_provider, model, and estimator. This modular architecture ensures flexibility, scalability, and ease of use across diverse ITS applications.
- **`Data provider`**: it aims to process the dataset with diverse features and protocols to provide appropriate inputs for different ITS tasks. It inherits from the base class `data_provider/Base_Dataset`, which standardizes data handling procedures across the framework. Each dataset has a unique data_provider class, while each ITS task to execute corresponds to a specific in-class method within the data_provider class. 
- **`Model`**: it serves as the encoder to generate representations from the input data provided by Data provider. Leveraging advanced machine learning algorithms, the model captures intricate dependencies inherent in ITS data. Notably, the models are generally designed to be task-agnostic, allowing it to be used across different tasks.
- **`Estimator`**: it wraps the data_provider, model, and decoder for a given task, defining the training and evaluation protocols. The decoder, which is cascaded from the model, transforms the encoded representations into task-specific outputs. For example, in the SS task, the decoder is an affine layer that transforms the representation $z$ into a real-valued estimate of the quality variable $y$; in the PdM task, the decoder is an affine layer followed by a softmax layer that transforms $z$ into a probability vector, indicating the likelihood of anomalies occurring at corresponding time stamps.



## ❖ Installation

We will release a package on PyPI and anaconda very soon. Before that, you can install PyITS by cloning it.
``` bash
git clone https://github.com/Master-PLC/PyITS.git
```


## ❖ Contribution

We warmly invite you to contribute to PyITS. By committing your code:

1. You can make your works readily available for PyITS users to run, which helps your work gain more recognition and credits.
2. You will be listed as a PyITS contributor and a volunteer developer.
3. Your contributions will be highlighted in PyITS release notes.

Refer to the `data_provider/data_generator.py`, `models/Transformer.py` and `estimator/foundation/soft_sensor_estimator.py` to see how to include your dataset, model and estimator in PyITS.


## ❖ Acknowledgement