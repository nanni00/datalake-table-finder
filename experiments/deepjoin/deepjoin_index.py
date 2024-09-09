import multiprocessing as mp
from time import time

import faiss
import numpy as np
from datasets import Dataset

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.training_args import BatchSamplers

from tools.utils.parallel_worker import chunks
from tools.utils.datalake import SimpleDataLakeHelper
from tools.utils.misc import is_valid_table

model = SentenceTransformer.load('/data4/nanni/tesi-magistrale/models/mpnet-base-all-nli-triplet/final')


dlh = SimpleDataLakeHelper('mongodb', 'wikiturlsnap', 'standard')



model.encode_multi_process()


