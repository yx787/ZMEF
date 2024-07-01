# ZMEF: Zero-shot Multi-Exposure Image Fusion
##  ZMEF: Zero-shot Multi-Exposure Image Fusion
By Xuziqian Yang, Xiao Tan, Yongbin Liu, Huaian Chen, Yi Jin, Haoxuan Wang, Enhong Chen

### Highlights
- **the first zero-shot method for MEF task**  a simple yet effective model that can learn a high-quality image from a given sequence containing two or more different exposed images, eliminating the requirements of any prepared training data.
- **self-supervised illmination loss**:  consisting of intra-frame and inter-frame losses is deliberately designed. 
- **State of the art**

## Dependencies and Installation

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch](https://pytorch.org/)
- CUDA Version: 11.6

## Set

The test parameters can be adjusted in **test.yaml**, which can be found in **./config** folder.

## Test
  ```
  python main.py
  
  ```