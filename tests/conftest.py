"""
Configure sys.path et sys.modules avant tout import de modules du projet.
Nécessaire car un paquet `tools` tiers est installé dans l'environnement
et masquerait sinon src/tools/.
"""
import sys
import os

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))

# Toujours insérer en tête pour prendre la priorité sur les paquets installés
sys.path.insert(0, SRC_PATH)

# Si 'tools' est déjà en cache depuis le mauvais emplacement, on le purge
if "tools" in sys.modules:
    cached_file = getattr(sys.modules["tools"], "__file__", "") or ""
    if SRC_PATH not in cached_file:
        for key in list(sys.modules):
            if key == "tools" or key.startswith("tools."):
                del sys.modules[key]
