import sys
import os
import threading
import webbrowser
import time

sys.path.append(os.path.abspath("."))

def open_browser():
    time.sleep(3)  # Wait for server to start
    webbrowser.get("windows-default").open("http://localhost:8501")

threading.Thread(target=open_browser).start()
os.system("streamlit run ui/app.py")