import subprocess
import time

NUM_CLIENTS = 3

# Start server
subprocess.Popen(["python", "server.py"])
time.sleep(2) 

# Start multiple clients
for client_id in range(1, NUM_CLIENTS + 1):
    subprocess.Popen(["python", "client.py", str(client_id)])
