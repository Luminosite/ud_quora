import Queue
import threading
import time


class WorkThread (threading.Thread):
    def __init__(self, func, data, tag, index, q, lock):
        threading.Thread.__init__(self)
        self.func = func
        self.data = data
        self.tag = tag
        self.index = index
        self.q = q
        self.lock = lock
        
    def run(self):
        print("thread %d for %s starts" % (self.index, self.tag))
        ret = self.func(data)
        self.lock.acquire()
        self.q.put((self.index, ret))
        self.lock.release()
        print("thread %d for %s finished" % (self.index, self.tag))
        
        
def process_data(func, data, tag, n):
    queueLock = threading.Lock()
    workQueue = Queue.Queue(n)
    
    l = data.shape[0]
    block_len = (l-1)/n + 1
    
    for i in range(n):
        d = data[i*block_len: (i+1)*block_len]
        t = WorkThread(func, d, tag, i, workQueue, queueLock)
        t.start()
    
    result = None
    finish = False
    while not finish:
        queueLock.acquire()
        if workQueue.qsize() == n:
            ret = list(workQueue.queue)
            ret = sorted(ret, key=lambda x: x[0])
            ret = [x[1] for x in ret]
            result = pd.concat(ret, axis=0).reset_index(drop=True)
            finish = True
        else:
            time.sleep(3)
        queueLock.release()
    
    return result
