import multiprocessing as mp

MAX_THREADS = 16

THREADS = min(MAX_THREADS, mp.cpu_count())

