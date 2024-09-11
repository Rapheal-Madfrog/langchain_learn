# -*- coding: utf-8 -*-
# author = 'ty'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
