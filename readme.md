# Parallel Prediction Model Based on Multi-Level Attention Mechanism

Parallel prediction model based on Transformer-BiGRUGlobalAttention-CrossAttention! ● Based on the multi-head attention mechanism of Transformer and parallel computing capability!  
● Based on the GlobalAttention optimization of BiGRU!  
● Based on cross attention to fuse spatiotemporal features!  
● Environment framework: Python 3.9, PyTorch 1.8 and above can run  
● Using objects: Paper requirements, graduation design requirements  
● Code guarantee: The code is detailed with comments, ready to run and pass.  
50 epochs, training error is very small, multi-variable feature sequence Transformer-BiGRUGlobalAttention-CrossAttention parallel fusion network model prediction effect is significant, the model can fully extract the spatial and temporal features of time series, convergence speed is fast, performance is excellent, prediction accuracy is high, can extract important features for model prediction from the sequence spatiotemporal features, effect is obvious! Innovation degree also!!!  
Environment: Python 3.9, PyTorch 1.8 and above can run  
Model innovation points, students who need a graduation design or a small paper must see this, rich modules, high innovation degree, excellent performance!
Innovation points:  
1. Use Transformer to extract the long-term dependencies in the sequence's temporal features, adopting a parallel structure, speed up the model's training and inference speed, improve the model's ability to perceive key information;
2. By employing a bidirectional gated recurrent unit (BiGRU) to model sequences from both forward and backward directions simultaneously, we aim to better capture dependencies within the sequence while applying a global attention mechanism GlobalAttention to weight the output of the BiGRU, enabling the model to focus on the most critical parts of the sequence, thereby improving prediction performance;
3. Leveraging cross-attention for parallel network integration of spatiotemporal features, which allows for consideration of both temporal and spatial relationships simultaneously, thus capturing more features in spatiotemporal sequence data, enhancing feature representation capabilities to achieve high-precision predictions.
4. The product has demonstrated high-precision prediction performance across multiple datasets!
5. Note: This product also includes accompanying models explanations (for learning about network structures) and parameter tuning explanations!
6. Additionally, the code is accompanied by beautifully designed drawings and visualizations of deep learning prediction model feature importance!
7. Model training visualization graph:
8. Multi-feature contribution analysis visualization graph:
9. This analysis code is our team's original creation, demonstrating how to utilize a well-trained deep learning model to visualize feature importance (contribution) in multi-feature prediction tasks! (This can also be used for other deep learning models for feature importance visualization, making the code highly applicable.)
10. 
Preface
This article introduces a parallel prediction model based on Transformer-BiGRUGlobalAttention-CrossAttention, based on the previous introduction to the power transformer (Appendix for dataset), to improve the prediction performance of time series data. A detailed introduction to the power transformer dataset can be found in the following text:
Power Transformer Dataset Introduction and Preprocessing
The overall structure of the model is as follows: multi-feature variable time series data first undergoes a Transformer encoder layer based on multi-head attention to extract long-term dependencies, while data also passes through a BiGRU network optimized by GlobalAttention to extract global temporal features. A cross-attention mechanism is used for feature fusion, and through the calculation of attention weights, the model focuses more on important features before enhancing them for fusion. Finally, it goes through a fully connected layer for high-precision prediction.
Branch one: The Transformer encoder model based on multi-head attention captures the dependencies between different positions in the sequence through attention mechanisms, which can capture the global contextual information in the sequence.
Branch two: Multi-feature sequence data simultaneously passes through a BiGRU network optimized by GlobalAttention. GlobalAttention is a mechanism used to enhance the model's focus on different parts of the input sequence. In the BiGRU model, the global attention mechanism helps the model better concentrate on the most relevant parts of the input sequence, thereby improving the model's performance and generalization ability. At each time step, the global attention mechanism calculates a weight vector that represents the model's focus on different parts of the input sequence. These weights are then applied to the feature representation output of the BiGRU, weighting all features at each position to allow the model to focus more specifically on important temporal features in the sequence, enhancing its perception capability of multi-feature sequence temporal features;
Parallel prediction:
The model adopts a parallel structure, capable of predicting multiple target steps simultaneously.
Parallel prediction can accelerate the training and inference speed of the model and make full use of the information in the time series data, improving prediction performance.
Cross-attention mechanism feature fusion: Using the cross attention mechanism to fuse spatial and temporal features, it is possible to calculate attention weights, learn the correlation between different positions within the spatiotemporal features, which can better capture the characteristics of spatiotemporal sequences in data, thereby enhancing model performance and generalization capabilities.
Global attention mechanism:
Global Attention Mechanism
2. Multi-feature variable dataset preparation and preprocessing:
2.1 Importing data
2.2 Creating datasets
2.3 Creating datasets with classification labels
3 Cross attention mechanism:
3.1 Concept of Cross attention
In the Transformer architecture, an attention mechanism that mixes two different embedding sequences is introduced.
Both sequences must have the same dimensionality.
The two sequences can be of different patterns (e.g., text, sound, image).
One sequence serves as the input Q, defining the output sequence length, while the other provides input K&V.
3.2 Cross-attention algorithm
With two sequences S1 and S2:
Calculate S1's K, V
Calculate S2's Q
Based on K and Q, compute the attention matrix
Apply V to the attention matrix
Output sequence length consistent with S2
During the fusion process, we use the temporal features through the Transformer as the query sequence, and the global spatial features optimized by GlobalAttention as key-value pairs. By calculating the attention weights between the query sequence and the key-value pairs, we can model the degree of association between different features.
4 Multi-level attention mechanism-based parallel high-precision prediction model:
4.1 Definition
Network model
\t\tNote: The input data shape is [64, 7, 7], batch_size=64. The number 7 represents the sequence length (the sliding window takes values), and the dimension of 7 dimensions represents the dimensions of 7 variables.
\t  
4.2 Set parameters and train the model
50 epochs, training error is extremely small. A multi-variable feature sequence Transformer-BiGRU Global Attention-Cross Attention parallel integration network model is used for prediction. The model can fully extract the spatial and temporal features of time series, has a fast convergence speed, superior performance, high prediction accuracy, and can extract important features from the sequence's spatiotemporal characteristics, with significant effects!
\tNote: Adjust parameters.
\t\tIt is appropriate to increase the number of Transformer encoder layers and hidden layer dimensions, as well as the number of multi-head attention heads, and fine-tune the learning rate.
Adjust the number of layers in the BiGRU and the number of neurons per layer, increasing more epochs (note to prevent overfitting).
\tThe sliding window length can be changed (set the appropriate window length).
5 Model Evaluation and Visualization
5.1 Results Visualization
5.2 Model Evaluation
\t5.3 Feature Visualization
\n        \n  
Previously Featured Content:
Time Series Forecasting: Analysis and comparison of LSTM, ARIMA, Holt-Winters, SARIMA models
Wind Speed Prediction (Part 1) Data Set Introduction and Preprocessing
Wind Speed Prediction (II) EMD-LSTM Model Based on Pytorch
Wind Speed Prediction (III) EMD-LSTM-Attention Model
Wind Speed Prediction (IV) EMD-Transformer Model Based on Pytorch
Wind Speed Prediction (5) EMD-CNN-LSTM Model Based on Pytorch
Wind Speed Prediction (VI) - EMD-CNN-GRU Parallel Model
CEEMDAN + Combination Forecasting Model (BiLSTM-Attention + ARIMA)
CEEMDAN + Combination Forecasting Model (CNN-LSTM + ARIMA)
CEEMDAN + Combination Forecasting Model (Transformer - BiLSTM + ARIMA)
CEEMDAN + Combination Forecasting Model (CNN-Transformer + ARIMA)
Multi-Feature Variable Sequence Forecasting (I) - CNN-LSTM Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (II) - CNN-LSTM-Attention Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (III) - CNN-Transformer Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (IV) Transformer-BiLSTM Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (V) CEEMDAN+CNN-LSTM Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (VI) CEEMDAN+CNN-Transformer Wind Speed Forecasting Model
Predictive Model Based on the Particle Swarm Optimization Algorithm SSA - CEEMDAN-BiLSTM-Attention
Predictive Model Based on the Particle Swarm Optimization Algorithm SSA - CEEMDAN-Transformer-BiGRU
Superior Predictive Model: Secondary Decomposition-Combination Forecasting
Multi-Feature Variable Sequence Forecasting (VII) CEEMDAN+Transformer-BiLSTM Forecasting Model
Multi-Feature Variable Sequence Forecasting (VIII) Predictive Model Based on the Particle Swarm Optimization Algorithm for CEEMDAN-SSA-BiLSTM
Multi-Feature Variable Sequence Forecasting (11) Predictive Model Based on Pytorch's TCN-GRU
VMD + CEEMDAN Secondary Decomposition, CNN-LSTM Prediction Model
VMD + CEEMDAN Secondary Decomposition, CNN-Transformer Prediction Model
Prediction Model Based on the Pigeon Optimization Algorithm SSA - Complete Code Kit
Superior Combination Prediction Model!
END
\t2. Friends who like it can click to follow, give the article a "like", share in the circle or discuss in the group.

## Images

[ | Multi-level Attention-based Parallel Prediction Model](01.png)
[ | Multi-level Attention-based Parallel Prediction Model](02.png)
[ | Multi-level Attention-based Parallel Prediction Model](03.png)
[ | Multi-level Attention-based Parallel Prediction Model](04.png)
[ | Multi-level Attention-based Parallel Prediction Model](05.png)
[ | Multi-level Attention-based Parallel Prediction Model](06.png)
[ | Multi-level Attention-based Parallel Prediction Model](07.png)
[ | Multi-level Attention-based Parallel Prediction Model](08.png)
[ | Multi-level Attention-based Parallel Prediction Model](09.png)
[ | Multi-level Attention-based Parallel Prediction Model](10.png)
[ | Multi-level Attention-based Parallel Prediction Model](11.png)
[ | Multi-level Attention-based Parallel Prediction Model](12.png)
[ | Multi-level Attention-based Parallel Prediction Model](13.png)
[ | Multi-level Attention-based Parallel Prediction Model](14.png)
[ | Multi-level Attention-based Parallel Prediction Model](15.png)
[ | Multi-level Attention-based Parallel Prediction Model](16.png)
[ | Multi-level Attention-based Parallel Prediction Model](17.png)
[ | Multi-level Attention-based Parallel Prediction Model](18.png)
[ | Multi-level Attention-based Parallel Prediction Model](19.png)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
