# Solutions et Corrections Proposées

## 🔧 Corrections pour les Failles CRITIQUES

### FAILLE #2 & #3: Gestion d'Exception et Retours Incohérents dans `bounds.py`

**AVANT**:
```python
def compute_bounds_data(network, x, epsilon, n, K, method: str = "IBP", norm : str = "Linf"):
    """
    Compute the  L and U

    Args:
        method (str): The method to compute the bounds (CROWN, IBP, Linear, etc.).
    """
    print(f"STUDY : Computing bounds with method: {method} ...")
    print("epsilon : ", epsilon)
    L = [[-np.inf] * n[k] for k in range(K + 1)]
    U = [[np.inf] * n[k] for k in range(K + 1)]

    if method == "GREAT_BOUNDS":
        L[0] = [max(L[0][j], 0) for j in range(len(L[0]))]
        return  # ❌ RETOUR VIDE!

    if not torch.is_tensor(x):
        x = torch.Tensor(x)

    x = x.type(torch.float).view(-1).unsqueeze(0).to(device)
    # ... plus loin ...
    try:
        bounded_model = BoundedModule(
            network,
            zeros,
            bound_opts={"conv_mode": "patches"},
        )
        print("created BoundedModule")
    except Exception as e:
        print("Error creating BoundedModule:", e)
        return  # ❌ RETOUR VIDE!
```

**APRÈS**:
```python
import logging

logger = logging.getLogger(__name__)

def compute_bounds_data(network: torch.nn.Module, x, epsilon: float, n: list, K: int, 
                       method: str = "IBP", norm: str = "Linf") -> tuple:
    """
    Compute the lower bounds (L) and upper bounds (U) for network activations.

    Args:
        network (torch.nn.Module): Neural network model
        x (array-like): Input point to certify
        epsilon (float): Perturbation radius (must be >= 0)
        n (list): Number of neurons per layer
        K (int): Number of layers
        method (str): Bounds computation method (CROWN, IBP, Linear, alpha-CROWN)
        norm (str): Perturbation norm (Linf, L2)

    Returns:
        tuple: (L, U) where L and U are lists of layer bounds
        
    Raises:
        ValueError: If epsilon < 0 or K/n are invalid
        RuntimeError: If bounds computation fails
    """
    logger.debug(f"Computing bounds with method: {method}, epsilon: {epsilon}, norm: {norm}")
    
    # Validation des entrées
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    if K < 0 or not isinstance(K, int):
        raise ValueError(f"K must be a non-negative integer, got {K}")
    if len(n) != K + 1:
        raise ValueError(f"n must have {K+1} elements, got {len(n)}")
    if norm not in ["Linf", "L2"]:
        raise ValueError(f"norm must be 'Linf' or 'L2', got {norm}")

    L = [[-np.inf] * n[k] for k in range(K + 1)]
    U = [[np.inf] * n[k] for k in range(K + 1)]

    if method == "GREAT_BOUNDS":
        L[0] = [max(L[0][j], 0) for j in range(len(L[0]))]
        return L, U  # ✅ RETOUR CORRECT

    if not torch.is_tensor(x):
        x = torch.Tensor(x)

    x = x.type(torch.float).view(-1).unsqueeze(0).to(device)
    logger.debug(f"Input shape: {x.shape}, device: {x.device}")

    network = network.to(device)
    network.eval()

    zeros = torch.zeros_like(x).to(device)

    logger.debug("Creating BoundedModule...")
    try:
        bounded_model = BoundedModule(
            network,
            zeros,
            bound_opts={"conv_mode": "patches"},
        )
        logger.info("BoundedModule created successfully")
    except Exception as e:
        logger.error(f"Failed to create BoundedModule: {e}", exc_info=True)
        raise RuntimeError(f"BoundedModule creation failed: {e}") from e  # ✅ REMONTE L'ERREUR

    bounded_model.eval()

    if norm == "Linf":
        ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
    elif norm == "L2":
        ptb = PerturbationLpNorm(norm=2, eps=epsilon)
    else:
        raise NotImplementedError(f"Norm {norm} not implemented.")
    
    bounded_image = BoundedTensor(x, ptb)
    
    try:
        if method == "alpha-CROWN":
            lb, ub = bounded_model.compute_bounds(x=(bounded_image,), method=method)
        else:
            with torch.no_grad():
                lb, ub = bounded_model.compute_bounds(x=(bounded_image,), method=method)
    except Exception as e:
        logger.error(f"Bounds computation failed: {e}", exc_info=True)
        raise RuntimeError(f"Bounds computation failed with method {method}: {e}") from e

    intermediate_bounds = bounded_model.save_intermediate()
    intermediate_bounds_list = list(intermediate_bounds.keys())
    
    # ... rest of code ...
    
    L = round_list_depth_2(L)
    U = round_list_depth_2(U)
    
    return L, U  # ✅ RETOUR TOUJOURS COHÉRENT
```

---

### FAILLE #6: Dépendances Manquantes

**AVANT**:
```toml
[project]
name = "FastSDPCertification"
version = "0.1"
description = "Ton projet"
requires-python = ">=3.7"
dependencies = []
```

**APRÈS**:
```toml
[project]
name = "FastSDPCertification"
version = "0.1"
description = "Fast SDP-based Neural Network Certification"
requires-python = ">=3.8"
dependencies = [
    "torch>=1.12.0",
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "pydantic>=1.9.0",
    "PyYAML>=5.4.0",
    "pandas>=1.3.0",
    "matplotlib>=3.4.0",
    "torchvision>=0.13.0",
    "auto-LiRPA>=0.4.0",
    "gurobipy>=10.0.0",
    "Mosek>=10.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0.0",
    "black>=21.0.0",
    "pylint>=2.10.0",
    "mypy>=0.900",
]
```

---

### FAILLE #7: Chemins en Dur

**AVANT**:
```python
with open("weights_nn.txt", "w") as f:
    # ...
```

**APRÈS**:
```python
import os
from pathlib import Path

def get_output_path(filename: str) -> Path:
    """Get output path for files, creating directory if needed."""
    output_dir = Path(get_project_path("results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename

# Usage
output_file = get_output_path("weights_nn.txt")
try:
    with open(output_file, "w") as f:
        # ...
except IOError as e:
    logger.error(f"Failed to write to {output_file}: {e}")
    raise
```

---

### FAILLE #8: Validation Input

**AVANT**:
```python
def __init__(
    self,
    network: networks.ReLUNN,
    epsilon: float,
    norm: str,
    dataset: TensorDataset,
    **kwargs,
):
    self.network = network.to(device_ if kwargs.get("use_cuda", True) else "cpu")
    self.epsilon = epsilon
    self.norm = norm
    self.dataset = dataset
```

**APRÈS**:
```python
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Certification_Problem:
    VALID_NORMS = ["Linf", "L2", "L1"]
    
    def __init__(
        self,
        network: networks.ReLUNN,
        epsilon: float,
        norm: str,
        dataset: TensorDataset,
        use_cuda: bool = True,
        network_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        yaml_file: Optional[str] = None,
        models: Optional[list] = None,
    ):
        """Initialize the certification problem.
        
        Args:
            network: The neural network model
            epsilon: Perturbation bound (must be >= 0)
            norm: Perturbation norm type (Linf, L2, L1)
            dataset: TensorDataset for certification
            use_cuda: Whether to use CUDA device
            network_name: Optional network identifier
            dataset_name: Optional dataset identifier
            yaml_file: Optional YAML config file path
            models: Optional list of models to use
            
        Raises:
            ValueError: If parameters are invalid
            TypeError: If types are incorrect
        """
        # Validation
        if network is None:
            raise ValueError("network cannot be None")
        if not isinstance(epsilon, (int, float)) or epsilon < 0:
            raise ValueError(f"epsilon must be a non-negative number, got {epsilon}")
        if norm not in self.VALID_NORMS:
            raise ValueError(f"norm must be one of {self.VALID_NORMS}, got {norm}")
        if dataset is None or len(dataset) == 0:
            raise ValueError("dataset cannot be None or empty")
        
        logger.info(f"Initializing Certification Problem with epsilon={epsilon}, norm={norm}")
        
        self.network = network.to(torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu"))
        self.epsilon = epsilon
        self.norm = norm
        self.dataset = dataset
        self.models = models or []
        self.network_name = network_name or network.name
        self.dataset_name = dataset_name
        self.yaml_file = yaml_file
        
        logger.info("Certification Problem initialized successfully")
```

---

## 🔧 Corrections pour les Failles MAJEURES

### FAILLE #9: Logging Désactivé

**AVANT**:
```python
logger_mosek = logging.getLogger("Mosek_logger")
logger_mosek.setLevel(logging.DEBUG)
logger_mosek.propagate = False
handler = logging.FileHandler(get_project_path("results/Mosek_logger.log"))
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger_mosek.addHandler(handler)
```

**APRÈS**:
```python
import logging
import logging.config
from pathlib import Path

def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    log_dir = Path(get_project_path("results"))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("FastSDPCertification")
    logger.setLevel(level)
    
    # File handler
    fh = logging.FileHandler(log_dir / "fastsdp.log")
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# Usage
logger = setup_logging(level="DEBUG")
```

---

### FAILLE #16: Print Statements Debug

**AVANT**:
```python
print(f"STUDY : Computing bounds with method: {method} ...")
print("epsilon : ", epsilon)
print("x device : ", x.device)
print("x shape : ", x.shape)
print("network device : ", next(network.parameters()).device)
print("network is none : ", network is None)
```

**APRÈS**:
```python
import logging

logger = logging.getLogger(__name__)

# Configure based on environment
import os
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=log_level)

# Then use:
logger.debug(f"Computing bounds with method: {method}")
logger.debug(f"epsilon: {epsilon}")
logger.debug(f"Input device: {x.device}")
logger.debug(f"Input shape: {x.shape}")
logger.debug(f"Network device: {next(network.parameters()).device}")
```

---

## 📝 Fichier de Configuration Recommandé

**Créer `src/logging_config.py`**:
```python
"""Logging configuration for FastSDPCertification."""

import logging
import logging.config
from pathlib import Path
from typing import Optional

def setup_logging(
    log_dir: Optional[str] = None,
    level: str = "INFO",
    log_file: str = "fastsdp.log"
) -> logging.Logger:
    """
    Setup logging with both file and console handlers.
    
    Args:
        log_dir: Directory for log files (default: results/)
        level: Logging level
        log_file: Log file name
        
    Returns:
        Configured logger
    """
    if log_dir is None:
        log_dir = Path("results")
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.FileHandler",
                "level": level,
                "formatter": "detailed",
                "filename": str(log_dir / log_file)
            }
        },
        "root": {
            "level": level,
            "handlers": ["console", "file"]
        }
    }
    
    logging.config.dictConfig(logging_config)
    return logging.getLogger("FastSDPCertification")


if __name__ == "__main__":
    logger = setup_logging(level="DEBUG")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
```

---

## 🧹 Cleanup Script

**Créer `scripts/cleanup_code.py`**:
```python
"""Cleanup script to remove debug statements and commented code."""

import re
from pathlib import Path

def cleanup_debug_statements(file_path: Path) -> int:
    """Remove STUDY: print statements."""
    content = file_path.read_text()
    original = content
    
    # Remove STUDY print statements
    content = re.sub(r'^\s*print\s*\(\s*["\']STUDY\s*:.*?\)\s*$', '', content, flags=re.MULTILINE)
    
    # Count changes
    changed = content != original
    if changed:
        file_path.write_text(content)
        print(f"Cleaned: {file_path}")
    
    return 1 if changed else 0

def main():
    src_dir = Path("src")
    total = 0
    
    for py_file in src_dir.rglob("*.py"):
        total += cleanup_debug_statements(py_file)
    
    print(f"\nTotal files cleaned: {total}")

if __name__ == "__main__":
    main()
```

---

## ✅ Testing Recommendations

```python
# tests/test_bounds.py
import pytest
import torch
from src.bounds import compute_bounds_data

def test_compute_bounds_data_validation():
    """Test input validation."""
    network = MockNetwork()
    x = [1.0, 2.0]
    
    # Test negative epsilon
    with pytest.raises(ValueError, match="epsilon must be >= 0"):
        compute_bounds_data(network, x, epsilon=-1.0, n=[2, 3], K=1)
    
    # Test invalid norm
    with pytest.raises(ValueError, match="norm must be"):
        compute_bounds_data(network, x, epsilon=0.1, n=[2, 3], K=1, norm="invalid")
    
    # Test mismatched n and K
    with pytest.raises(ValueError, match="n must have"):
        compute_bounds_data(network, x, epsilon=0.1, n=[2], K=2)

def test_compute_bounds_data_return_type():
    """Test that function returns proper tuple."""
    network = MockNetwork()
    x = [1.0, 2.0]
    
    L, U = compute_bounds_data(network, x, epsilon=0.1, n=[2, 3], K=1)
    
    assert isinstance(L, list)
    assert isinstance(U, list)
    assert len(L) == 2
    assert len(U) == 2
```

---

## 🎯 Priorisation des Corrections

**Week 1**: Failles 1-4 (Erreurs critiques)  
**Week 2**: Failles 5-10 (Failles majeures)  
**Week 3**: Failles 11-19 (Nettoyage et maintenance)

