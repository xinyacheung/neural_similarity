
# [Adaptive stretching of representations across brain regions and deep learning model layers](https://www.biorxiv.org/content/10.1101/2023.12.01.569615v1)
Xin-Ya Zhang, Sebastian Bobadilla-Suarez, Xiaoliang Luo, Marilena Lemonari, Scott L. Brincat, Markus Siegel, Earl K. Miller, Bradley C. Love. bioRxiv.

#### This code repository includes the following documents
demo: Example codes for plotting the data figures in the main text;

data: cue images and electrode monkey spike data sample from [1];
> [1] Siegel, M., Buschman, T. J. & Miller, E. K. Cortical information flow during flexible sensorimotor decisions. Science 348, 1352–1355 (2015).
> 
neuralcode: process neuronal data from monkey spike data and get representational dissimilarity matrix (RDM);

NNmodel: the CNN-LSTM model example was fed the same movies as that presented to monkeys in the monkey experiments
and closely replicated their training procedure.

#### Software Dependencies

* Python (>= 3.8)
* NumPy (>= 1.21.2)
* Pytorch (>= 1.10.0)
* Pyspike (=0.8.0)
* Scikit-learn (>=1.0.2)
* Scipy (>= 1.7.3)
* Seaborn (>= 0.11.2)
* Pandas (>= 1.4.1)
* Matplotlib (>= 3.5.1)
* Torch-intermediate-layer-getter (0.1.post1)
