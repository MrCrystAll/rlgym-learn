import queue
import sys
import threading

def read(input_queue: queue.Queue, event: threading.Event):
    while not event.is_set():
        char = sys.stdin.read(1)
        if char:
            input_queue.put(char)

class STDINReader:
    def __init__(self) -> None:
        self.input_queue = queue.Queue()
        self.event = threading.Event()
        
        self.read_thread = threading.Thread(target=read, daemon=True, args=(self.input_queue, self.event))
        
    def stop(self):
        self.event.set()
        self.read_thread.join(timeout=1.0)
                
    def start_reading(self):
        self.read_thread.start()
        
    def getch(self):
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None