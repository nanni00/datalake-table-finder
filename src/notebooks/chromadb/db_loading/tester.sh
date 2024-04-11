#!/bin/bash

#                                       n_tables    batch_size      add_label   with_metadatas
python db_loading_two_collections.py   25000       500             False       False         
python db_loading_two_collections.py   25000       500             False       True          
python db_loading_two_collections.py   25000       2500            True        False           
python db_loading_two_collections.py   25000       2500            True        True            
python db_loading_two_collections.py   50000       500             False       False
python db_loading_two_collections.py   50000       500             True        True
python db_loading_two_collections.py   50000       2500            False       False
python db_loading_two_collections.py   50000       2500            True        True


