import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import ort_aot

ort_aot.aot.debug_ort_training_mode(Path("debug.onnx").resolve(strict=True))

