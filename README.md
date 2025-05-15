This project file contains:

   1. Requirements file:
        -requirements.txt
            **install requirements by pip install -r requirements.txt **

   2. Three baseline models:
    ## The three baseline model are from PAN challenge official document and were modified a bit
        **run them by: run xxx.py** 
        -backtranslation_baseline_clef25.py
        -baseline_delete_clef25.py
        -mt0_baseline_clef25.py

   3. Qwen3 based models with data preparation codes:
    ## The four Qwen3 based models are for our project
        -qwen3_14b_0shot.py
            **zero shot experiment with Qwen3-14B**
            **run by command: run qwen3_14b_0shot.py**

        -qwen3_14b_fewshot.py
            **few shot experiment with Qwen3-14B, randomly select 3 examples for each language as shots**
            **run by command: run qwen3_14b_fewshot.py**

        -qwen3_14b_rag.py
        -build_multi_index.py
            **train Qwen3-14B with RAG strategy**
            **before running Qwen3-14B model, prepare the embedding of traing data via: run build_multi_index.py**
            **run RAG model by command: run qwen3_14b_rag.py**
        
        -qwen3_14b_finetune_LoRA_peft.py
        -finetune_prepare.py
            **finetune Qwen3-14B with Low Rank Adaptation and peft strategy**
            **prepare the training data via: run finetune_prepare.py**
            **run finetuning model by command: run qwen3_14b_finetune_LpRA_peft.py**

   4. Dataset:
    ## the dataset is a document, de refers to German, en refers to English, es refers to Spanish and zh refers to Chinese
    ## xx_data.tsv: full dataset
    ## xx_train.tsv: training set(70%)
    ## xx_dev.tsv: development set(15%)
    ## xx_test.tsv: test set(15%)
        =input_data
            -de_data.tsv
            -de_train.tsv
            -de_dev.tsv
            -de_test.tsv
            -en_data.tsv
            -en_train.tsv
            -en_dev.tsv
            -en_test.tsv
            -es_data.tsv
            -es_train.tsv
            -es_dev.tsv
            -es_test.tsv
            -zh_data.tsv
            -zh_train.tsv
            -zh_dev.tsv
            -zh_test.tsv

   5. Evaluations:
    ## evaluation method is adopted from PAN challenges
        ==evaluation
            =metrics
                -similarity.py
                -toxity.py
                =fluency
                    -deberta_encoder.py
                    -xcomet.py
            -evaluate.py
            -utils.py
        **run the evaluation by command: run evaluate.py**
