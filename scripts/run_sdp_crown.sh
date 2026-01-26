
#!/bin/bash
set -e

NETWORK_NAME=$1
NOM_RUN=$2
EPSILON=$3

if [ -z "$NETWORK_NAME" ] || [ -z "$NOM_RUN" ]; then
  echo "Usage: $0 NETWORK_NAME NOM_RUN [EPSILON]"
  exit 1
fi

if [ -z "$EPSILON" ]; then
  case "$NETWORK_NAME" in
    6x100|9x100) EPSILON=0.026 ;;
    6x200|9x200) EPSILON=0.015 ;;
    MLP-ADV) EPSILON=0.1 ;;
    *) echo "Erreur: EPSILON non fourni."; exit 1 ;;
  esac
fi

if [[ "$NETWORK_NAME" =~ ^(6x100|6x200|9x100|9x200|MLP-ADV)$ ]]; then
  NETWORK_NAME="mnist-$NETWORK_NAME"
fi

SDP_CROWN_DIR="/share/homes/boyerma/SDP-CROWN"
if [ ! -d "$SDP_CROWN_DIR" ]; then
  echo "Erreur: Le dossier $SDP_CROWN_DIR n'existe pas !"
  exit 1
fi

cd "$SDP_CROWN_DIR"

DATE=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="/share/homes/boyerma/FastSDPCertification/results/benchmark/${NETWORK_NAME}_${EPSILON}/${DATE}_SDP-CROWN_${NOM_RUN}"
mkdir -p "$LOG_DIR"

echo "📁 Logs stockés dans : $LOG_DIR"

indexes_interval_mnist=(
  "0 56" "58 62" "64 74" "77 77" "81 86" "89 89"
  "92 93" "95 95" "100 100" "103 104" "106 107"
  "111 111" "115 117" "128 129" "135 135"
  "139 139" "148 148" "155 155" "165 165"
)

for interval in "${indexes_interval_mnist[@]}"; do
  read START END <<< "$interval"
  ((END++))
  LOG_FILE="$LOG_DIR/run_${START}_${END}.log"

  {
    echo "=============================="
    echo "Run: start=$START end=$END"
    echo "Début: $(date)"
    echo "=============================="

    python sdp_crown.py \
      --model "$NETWORK_NAME" \
      --radius "$EPSILON" \
      --start "$START" \
      --end "$END"

    echo "=============================="
    echo "Fin: $(date)"
    echo "=============================="
  } > "$LOG_FILE" 2>&1

  echo "✅ Terminé: start=$START end=$END"
done
