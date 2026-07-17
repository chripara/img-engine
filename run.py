import threading, warnings
from diffusers.utils import logging as diffusers_logging

# def _disable_warnings(self):
diffusers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*cannot run with `cpu` device.*")
warnings.filterwarnings("ignore", message=".*Overwriting.*in registry.*")
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="compel")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", message=".*Importing from timm.*")

from ui.ui import launch_ui
from app import create_app
app = create_app()

def run_flask():
    app.run(use_reloader=False)


if __name__ == '__main__':
    t = threading.Thread(target=run_flask)
    t.daemon = True
    t.start()

    launch_ui()
