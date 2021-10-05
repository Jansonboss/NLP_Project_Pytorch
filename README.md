# A Collection of NLP Project in Pytorch

Some important notes about NLP :relaxed:

The NLP Task is a lot differenct with CV, and NLP reserachers're working very hard to build feature extractor (`Architecture Engineering`) to make sure their architectures are capable of processing data with following features:

- Able to process `1d input data`.
	- eg. I love this movie. -> sentimental analysis

- Able to process `undefined Length` of input data.
	- eg. I love this movie.
	- eg. The movie I watched is shit.

- Able to capture `Long term dependency`.
	- eg. The **dog** is sitting there, and **he** is cute
	- **Dog** <---> **he**

- Able to recongnize `relaive position` between words is important.
	- `I owe u 10 buck` and `U owe me 10 bucks`is different 

- Able to process data at scale (Parallel Computing)


> 上面这几个特点请记清，一个特征抽取器是否适配问题领域的特点，有时候决定了它的成败，而很多模型改进的方向，其实就是改造得使得它更匹配领域问题的特性 like RNN, LSTM, Transformer etc. :no_mouth: :no_mouth: