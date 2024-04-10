#!/bin/bash

#                                       n_tables    batch_size      add_label   with_metadatas
#python analysis_index_size_scaling.py   25000       500             False       False         
python analysis_index_size_scaling.py   25000       500             False       True          
python analysis_index_size_scaling.py   25000       2500            True        False           
python analysis_index_size_scaling.py   25000       2500            True        True            


python analysis_index_size_scaling.py   50000       500             False       False
python analysis_index_size_scaling.py   50000       500             True        True
python analysis_index_size_scaling.py   50000       2500            False       False
python analysis_index_size_scaling.py   50000       2500            True        True


