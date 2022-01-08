import redis
from rq import Worker, Queue, Connection

listen = ['detector']

conn = redis.Redis(host='127.0.0.1', port=4444, decode_responses=False)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.work()