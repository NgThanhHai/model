#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import redis
from rq import Worker, Queue, Connection

listen = ['high', 'default', 'low']

redis_url = os.getenv('REDISTOGO_URL', 'redis://:p84a7f9ee51968d375603ad8f6b0e01622eec62c6979c0df621b4d2dc556740ea@ec2-3-231-201-215.compute-1.amazonaws.com:15789')

conn = redis.from_url(redis_url)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.work()


# In[ ]:





# In[ ]:




