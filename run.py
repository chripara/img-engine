from app import create_app
from ui.ui import launch_ui
import threading

app = create_app()

def run_flask():
    app.run(use_reloader=False)

if __name__ == '__main__':
    t = threading.Thread(target=run_flask)
    t.daemon = True
    t.start()
    
    launch_ui()

