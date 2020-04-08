import threading
import time

locka = threading.Lock()

a = 0
def add1():
    global a
    try:
        locka.acquire()
        tmp = a + 1
        time.sleep(0.2)
        a = tmp
    finally:
        locka.release()
    print('%s adds to 1: %d' % (threading.current_thread().getName(), a))

threads = [threading.Thread(name='t%d' % (i, ), target=add1) for i in range(10)]
[t.start() for t in threads]
