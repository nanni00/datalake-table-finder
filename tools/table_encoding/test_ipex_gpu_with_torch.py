#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:57:07 2024

@author: nanni
"""

import torch
import sentence_transformers
import intel_extension_for_pytorch as ipex


model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")

model = ipex.optimize(model, dtype=torch.float32)

print(model.device)