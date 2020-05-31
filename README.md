



# ml-toolbox

## List of machine learning libraries grouped by their usecase
<details>
<summary>Calculation Optimization</summary>

| Tool                                   | Description                                                                                                                                                       |
|:---------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| https://github.com/rapidsai/cudf       | cuDF is a GPU DataFrame library for loading, joining, aggregating, filtering, and otherwise manipulating data.                                                    |
| https://github.com/rapidsai/cuml       | cuML enables data scientists, researchers, and software engineers to run traditional tabular ML tasks on GPUs without going into the details of CUDA programming. |
| https://github.com/cupy/cupy           | NumPy-like API accelerated with CUDA                                                                                                                              |
| https://github.com/modin-project/modin | Modin: Speed up your Pandas workflows by changing a single line of code                                                                                           |
| https://github.com/numba/numba         | A Just-In-Time Compiler for Numerical Functions in Python                                                                                                         |
| https://github.com/weld-project/weld   | High-performance runtime for data analytics applications                                                                                                          |
</details>


<details>
<summary>Click Through Rate Prediction</summary>

| Tool                                   | Description                                                                                                                                     |
|:---------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|
| https://github.com/shenweichen/DeepCTR | Easy-to-use,Modular and Extendible package of deep-learning based CTR models.                                                                   |
| https://github.com/aksnzhy/xlearn      | xLearn is a high performance, easy-to-use, and scalable machine learning package that contains linear model (LR), factorization machines (FM), and field-aware factorization machines (FFM), all of which can be used to solve large-scale machine learning problems                           |
</details>


<details>
<summary>Computer Vision</summary>

| Tool                                     | Description                                                                                               |
|:-----------------------------------------|:----------------------------------------------------------------------------------------------------------|
| https://github.com/kornia/kornia/        | Kornia is a differentiable computer vision library for PyTorch.                                           |
| https://github.com/opencv/opencv         | Open Source Computer Vision Library                                                                       |
| https://github.com/madmaze/pytesseract   | A Python wrapper for Google Tesseract OCR Engine                                                          |
| https://github.com/sirfz/tesserocr       | A simple, Pillow-friendly, wrapper around the tesseract-ocr API for Optical Character Recognition (OCR).  |
| https://github.com/sightmachine/SimpleCV | SimpleCV is a framework for Open Source Machine Vision, using OpenCV and the Python programming language. |
</details>


<details>
<summary>Explainable AI</summary>

| Tool                                                   | Description                                                                                                                                                                                                                   |
|:-------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| https://github.com/pytorch/captum                      | Captum is a model interpretability and understanding library for PyTorch.                                                                                                                                                     |
| https://github.com/yosinski/deep-visualization-toolbox | Deep Visualization toolbox is an open source software tool that lets you probe DNNs by feeding them an image (or a live webcam feed) and watching the reaction of every neuron.                                               |
| https://github.com/TeamHG-Memex/eli5                   | ELI5 is a Python package which helps to debug machine learning classifiers and explain their predictions.                                                                                                                     |
| https://github.com/IBM/AIX360/                         | The AI Explainability 360 toolkit is an open-source library that supports interpretability and explainability of datasets and machine learning models.                                                                        |
| https://github.com/IBM/AIF360                          | The AI Fairness 360 toolkit is an extensible open-source library containg techniques developed by the research community to help detect and mitigate bias in machine learning models throughout the AI application lifecycle. |
| https://github.com/albermax/innvestigate               | This tool provides a common interface and out-of-the-box implementation for many analysis methods.                                                                                                                            |
| https://github.com/raghakot/keras-vis                  | keras-vis is a high-level toolkit for visualizing and debugging your trained keras neural net models.                                                                                                                         |
| https://github.com/marcotcr/lime                       | This project is about explaining what machine learning classifiers (or models) are doing and currently support explaining individual predictions for text classifiers or classifiers that act on                              |
|                                                        | tables (numpy arrays of numerical or categorical data) or images                                                                                                                                                              |
| https://github.com/interpretml/interpret               | InterpretML is an open-source python package for training interpretable machine learning models and explaining blackbox systems.                                                                                              |
| https://github.com/mindsdb/mindsdb                     | MindsDB is an Explainable AutoML framework for developers built on top of Pytorch that enables you to build, train and test state of the art ML models in as simple as one line of code.                                      |
| https://github.com/slundberg/shap                      | SHAP is a game theoretic approach to explain the output of any machine learning model                                                                                                                                         |
| https://github.com/tensorflow/cleverhans               | An adversarial example library for constructing attacks, building defenses, and benchmarking both                                                                                                                             |
| https://github.com/tensorflow/lucid                    | A collection of infrastructure and tools for research in neural network interpretability.                                                                                                                                     |
| https://github.com/tensorflow/model-analysis           | TensorFlow Model Analysis (TFMA) is a library for evaluating TensorFlow models that allows users to evaluate their models on large amounts of data in a distributed manner, using the same metrics defined in their trainer.  |
| https://github.com/andosa/treeinterpreter              | Package for interpreting scikit-learn's decision tree and random forest predictions.                                                                                                                                          |
</details>


<details>
<summary>Feature Engineering & Auto ML</summary>

| Tool                                          | Description                                                                                                                                                                                        |
|:----------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| https://github.com/blue-yonder/tsfresh        | Automatic extraction of relevant features from time series                                                                                                                                         |
| https://epistasislab.github.io/tpot/          | TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.                                                                              |
| https://github.com/rsteca/sklearn-deap        | It uses evolutionary algorithms instead of gridsearch in scikit-learn.                                                                                                                             |
| https://github.com/minimaxir/automl-gs        | Provide an input CSV and a target field to predict, generate a model + code to run it.                                                                                                             |
| https://automl.github.io/auto-sklearn/master/ | auto-sklearn frees a machine learning user from algorithm selection and hyperparameter tuning as it leverages recent advantages in Bayesian optimization, meta-learning and ensemble construction. |
</details>


<details>
<summary>Model Deployment</summary>

| Tool                                    | Description                                                                                                                                            |
|:----------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
| https://github.com/ucbrise/clipper      | A low-latency prediction-serving system                                                                                                                |
| https://github.com/kubeflow/kubeflow    | Machine Learning Toolkit for Kubernetes                                                                                                                |
| https://github.com/combust/mleap        | MLeap allows data scientists and engineers to deploy machine learning pipelines from Spark and Scikit-learn to a portable format and execution engine. |
| https://github.com/Microsoft/pai        | OpenPAI is an open source platform that provides complete AI model training and resource management capabilities                                       |
| https://github.com/SeldonIO/seldon-core | A framework to deploy, manage and scale your production machine learning to thousands of models                                                        |
| https://github.com/tensorflow/serving   | A flexible, high-performance serving system for machine learning models                                                                                |
| https://github.com/jolibrain/deepdetect | Deep Learning API and Server in C++11 support for Caffe, Caffe2, PyTorch,TensorRT, Dlib, NCNN, Tensorflow, XGBoost and TSNE                            |
</details>


<details>
<summary>Model and Data Versioning</summary>

| Tool                                       | Description                                                                                                                                                                              |
|:-------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| https://github.com/catalyst-team/catalyst  | PyTorch framework for Deep Learning research and development which was developed with a focus on reproducibility, fast experimentation and code/ideas reusing.                           |
| https://github.com/d6t/d6tflow             | d6tflow is a python library which makes building complex data science workflows easy, fast and intuitive.                                                                                |
| https://github.com/iterative/dvc           | Data Version Control | Git for Data & Models                                                                                                                                             |
| https://github.com/quantumblacklabs/kedro/ | A Python library that implements software engineering best-practice for data and ML pipelines.                                                                                           |
| https://github.com/mlflow/mlflow           | MLflow is a platform to streamline machine learning development, including tracking experiments, packaging code into reproducible runs, and sharing and deploying models.                |
| https://github.com/VertaAI/modeldb/        | ModelDB is an open-source system to version machine learning models including their ingredients code, data, config, and environment and to track ML metadata across the model lifecycle. |
| https://github.com/pachyderm/pachyderm     | Pachyderm: Data Versioning, Data Pipelines, and Data Lineage                                                                                                                             |
| https://github.com/polyaxon/polyaxon       | A platform for reproducible and scalable machine learning and deep learning on kubernetes                                                                                                |
| https://github.com/IDSIA/sacred            | Sacred is a tool to help you configure, organize, log and reproduce experiments                                                                                                          |
| https://github.com/allegroai/trains        | TRAINS - Auto-Magical Experiment Manager & Version Control for AI - NOW WITH AUTO-MAGICAL DEVOPS!                                                                                        |
</details>


<details>
<summary>Natural Language Processing</summary>

| Tool                                                     | Description                                                                                                                                                                                                              |
|:---------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| https://www.nltk.org/                                    | NLTK is a leading platform for building Python programs to work with human language data.                                                                                                                                |
| https://github.com/clips/pattern                         | Web mining module for Python, with tools for scraping, natural language processing, machine learning, network analysis and visualization.                                                                                |
| https://github.com/machinalis/quepy                      | A python framework to transform natural language questions to queries in a database query language.                                                                                                                      |
| https://github.com/sloria/TextBlob/                      | It provides a simple API for diving into common natural language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more.               |
| https://github.com/machinalis/yalign                     | Yalign is a tool for extracting parallel sentences from comparable corpora.                                                                                                                                              |
| https://github.com/columbia-applied-data-science/rosetta | Tools for data science with a focus on text processing.                                                                                                                                                                  |
| https://github.com/proycon/pynlpl                        | It can be used for basic tasks such as the extraction of n-grams and frequency lists, and to build simple language model.                                                                                                |
| https://github.com/sergioburdisso/pyss3                  | Python package that implements a novel text classifier (SS3) with visualizations tools for Explainable Artificial Intelligence (XAI)                                                                                     |
| https://github.com/explosion/spaCy                       | spaCy is a library for advanced Natural Language Processing built on the very latest research, and was designed from day one to be used in real products.                                                                |
| https://github.com/seatgeek/fuzzywuzzy                   | Fuzzy is a string matching tool that uses Levenshtein Distance to calculate the differences between sequences in a simple-to-use package.                                                                                |
| https://github.com/jamesturk/jellyfish                   | Jellyfish is a python library for doing approximate and phonetic matching of strings.                                                                                                                                    |
| https://github.com/chartbeat-labs/textacy                | textacy is a Python library for performing a variety of natural language processing (NLP) tasks, built on the high-performance spaCy library.                                                                            |
| https://github.com/aflc/editdistance                     | Fast implementation of the edit distance (Levenshtein distance).                                                                                                                                                         |
| https://github.com/dasmith/stanford-corenlp-python       | Python wrapper for Stanford University's NLP group's Java-based CoreNLP tools.                                                                                                                                           |
| https://github.com/cltk/cltk                             | The Classical Language Toolkit (CLTK) offers natural language processing (NLP) support for the languages of Ancient, Classical, and Medieval Eurasia.                                                                    |
| https://github.com/RasaHQ/rasa                           | Rasa is an open source machine learning framework to automate text-and voice-based conversations.                                                                                                                        |
| https://github.com/aboSamoor/polyglot                    | Polyglot is a natural language pipeline that supports massive multilingual applications.                                                                                                                                 |
| https://github.com/facebookresearch/DrQA                 | DrQA is a system for reading comprehension applied to open-domain question answering which is targeted at the task of "machine reading at scale" (MRS).                                                                  |
| https://github.com/dedupeio/dedupe                       | A python library for accurate and scalable fuzzy matching, record deduplication and entity-resolution.                                                                                                                   |
| https://github.com/snipsco/snips-nlu                     | It is a Python library that allows to extract structured information from sentences written in natural language.                                                                                                         |
| https://github.com/Franck-Dernoncourt/NeuroNER           | NeuroNER is a program that performs named-entity recognition (NER).                                                                                                                                                      |
| https://github.com/deepmipt/DeepPavlov/                  | DeepPavlov is an open-source conversational AI library built on TensorFlow and Keras, designed for the development of production ready chat-bots and complex conversational systems and                                  |
|                                                          | support research in the area of NLP and, particularly, of dialog systems.                                                                                                                                                |
| https://github.com/bigartm/bigartm                       | The state-of-the-art platform for topic modeling.                                                                                                                                                                        |
| https://github.com/EducationalTestingService/python-zpar | python-zpar is a python wrapper around the ZPar parser which is a statistical natural language parser, which performs syntactic analysis tasks including word segmentation,                                              |
|                                                          | part-of-speech tagging and parsing.                                                                                                                                                                                      |
| https://github.com/salesforce/ctrl                       | CTRL, a 1.6 billion-parameter conditional transformer language model, trained to condition on control codes that specify domain, subdomain, entities, relationships between entities, dates, and task-specific behavior. |
| https://github.com/facebookresearch/XLM                  | PyTorch original implementation of Cross-lingual Language Model Pretraining.                                                                                                                                             |
| https://github.com/flairNLP/flair                        | A very simple framework for state-of-the-art Natural Language Processing (NLP)                                                                                                                                           |
| https://github.com/github/semantic                       | semantic is a Haskell library and command line tool for parsing, analyzing, and comparing source code.                                                                                                                   |
| https://github.com/dmlc/gluon-nlp                        | GluonNLP is a toolkit that enables easy text preprocessing, datasets loading and neural models building to help you speed up your Natural Language Processing (NLP) research.                                            |
| https://github.com/gnes-ai/gnes                          | GNES is Generic Neural Elastic Search, a cloud-native semantic search system based on deep neural network.                                                                                                               |
| https://github.com/rowanz/grover                         | Grover is a model for Neural Fake News -- both generation and detection.                                                                                                                                                 |
| https://github.com/BrikerMan/Kashgari                    | Kashgari is a Production-ready NLP Transfer learning framework for text-labeling and text-classification, includes Word2Vec, BERT, and GPT2 Language Embedding.                                                          |
| https://github.com/explosion/sense2vec                   | sense2vec is a nice twist on word2vec that lets you learn more interesting and detailed word vectors.                                                                                                                    |
| https://github.com/snorkel-team/snorkel                  | A system for quickly generating training data with weak supervision                                                                                                                                                      |
| https://github.com/tensorflow/lingvo                     | Lingvo is a framework for building neural networks in Tensorflow, particularly sequence models.                                                                                                                          |
| https://github.com/vkcom/youtokentome                    | Unsupervised text tokenizer focused on computational efficiency                                                                                                                                                          |
| https://github.com/huggingface/transformers              | It provides state-of-the-art general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet, CTRL...) for Natural Language Understanding (NLU) and                                                          |
|                                                          | Natural Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between TensorFlow 2.0 and PyTorch.                                                                        |
| https://github.com/facebookresearch/wav2letter           | wav2letter++ is a fast, open source speech processing toolkit from the Speech team at Facebook AI Research built to facilitate research in end-to-end models for speech recognition                                      |
</details>


<details>
<summary>Recommendation System</summary>

| Tool                                       | Description                                                                                                                                   |
|:-------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------|
| https://github.com/maciejkula/spotlight    | Spotlight uses PyTorch to build both deep and shallow recommender models.                                                                     |
| https://github.com/cheungdaven/DeepRec     | An Open-source Toolkit for Deep Learning based Recommendation with Tensorflow.                                                                |
| https://github.com/NicolasHug/Surprise     | A Python scikit for building and analyzing recommender systems                                                                                |
| https://github.com/lyst/lightfm            | A Python implementation of LightFM, a hybrid recommendation algorithm.                                                                        |
| https://github.com/ocelma/python-recsys    | A python library for implementing a recommender system                                                                                        |
| https://github.com/jfkirk/tensorrec        | TensorRec is a Python recommendation system that allows you to quickly develop recommendation algorithms and customize them using TensorFlow. |
| https://github.com/caserec/CaseRecommender | Case Recommender is a Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback.         |
| https://github.com/benfred/implicit        | Fast Python Collaborative Filtering for Implicit Feedback Datasets                                                                            |
| https://github.com/ibayer/fastFM           | fastFM: A Library for Factorization Machines                                                                                                  |
</details>


<details>
<summary>Reinforcement Learning</summary>

| Tool                                     | Description                                                                                                                                                                                                                 |
|:-----------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| https://github.com/deepmind/lab          | DeepMind Lab provides a suite of challenging 3D navigation and puzzle-solving tasks for learning agents and is generally used as a testbed for research in artificial intelligence, especially deep reinforcement learning. |
| https://github.com/openai/gym            | A toolkit for developing and comparing reinforcement learning algorithms.                                                                                                                                                   |
| https://github.com/openai/retro          | Gym Retro lets you turn classic video games into Gym environments for reinforcement learning and comes with integrations for ~1000 games.                                                                                   |
| https://github.com/NervanaSystems/coach  | Coach is a python reinforcement learning framework containing implementation of many state-of-the-art algorithms.                                                                                                           |
| https://github.com/rlworkgroup/garage    | garage is a toolkit for developing and evaluating reinforcement learning algorithms, and an accompanying library of state-of-the-art implementations built using that toolkit.                                              |
| https://github.com/rlworkgroup/metaworld | Meta-World is an open-source simulated benchmark for meta-reinforcement learning and multi-task learning consisting of 50 distinct robotic manipulation tasks.                                                              |
</details>


<details>
<summary>Security & Privacy</summary>

| Tool                                         | Description                                                                                                                                                                       |
|:---------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| https://github.com/OpenMined/PySyft          | PySyft is a Python library for secure and private Deep Learning.                                                                                                                  |
| https://github.com/SubstraFoundation/substra | Substra is a framework for traceable ML orchestration on decentralized sensitive data.                                                                                            |
| https://github.com/tensorflow/privacy        | It is a Python library that includes implementations of TensorFlow optimizers for training machine learning models with differential privacy.                                     |
| https://github.com/tf-encrypted/tf-encrypted | TF Encrypted is a framework for encrypted machine learning in TensorFlow that aims to make privacy-preserving machine learning readily available, without requiring expertise in  |
|                                              | cryptography, distributed systems, or high performance computing.                                                                                                                 |
</details>


<details>
<summary>Visualization</summary>

| Tool                                            | Description                                                                          |
|:------------------------------------------------|:-------------------------------------------------------------------------------------|
| https://github.com/bokeh/bokeh                  | Interactive Data Visualization in the browser, from Python                           |
| https://github.com/andrea-cuttone/geoplotlib    | geoplotlib is a python toolbox for visualizing geographical data and making maps     |
| https://github.com/ResidentMario/missingno      | Missing data visualization module for Python.                                        |
| https://github.com/finos/perspective            | Perspective is an interactive visualization component for large, real-time datasets. |
| https://github.com/plotly/dash                  | Analytical Web Apps for Python, R, and Julia.                                        |
| https://github.com/Kozea/pygal                  | PYthon svg GrAph plotting Library                                                    |
| https://github.com/mwaskom/seaborn              | Statistical data visualization using matplotlib                                      |
| https://github.com/streamlit/streamlit          | Streamlit — The fastest way to build custom ML tools                                 |
| https://github.com/DistrictDataLabs/yellowbrick | Visual analysis and diagnostic tools to facilitate machine learning model selection. |
</details>


