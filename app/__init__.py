import sys
from pathlib import Path

# Add both project root and app directory to Python path
PROJECT_ROOT = Path(__file__).parent.parent
APP_DIR = Path(__file__).parent

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(APP_DIR))

