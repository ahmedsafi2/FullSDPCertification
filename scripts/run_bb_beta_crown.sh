
#!/bin/bash
set -e

DATASET_NAME=$1
NETWORK_NAME=$2
NOM_RUN=$3
EPSILON=$4


if [ -z "$DATASET_NAME" ] || ([ "$DATASET_NAME" != "cifar10" ] && [ "$DATASET_NAME" != "mnist" ]); then
  echo "Usage: $0 DATASET_NAME NETWORK_NAME NOM_RUN [EPSILON]"
  echo "DATASET_NAME must be 'cifar10' or 'mnist'"
  exit 1
fi

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

alpha_beta_CROWN_DIR="/share/homes/boyerma/alpha-beta-CROWN/complete_verifier"
if [ ! -d "$alpha_beta_CROWN_DIR" ]; then
  echo "Erreur: Le dossier $alpha_beta_CROWN_DIR n'existe pas !"
  exit 1
fi

cd "$alpha_beta_CROWN_DIR"
DATE=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="/share/homes/boyerma/FastSDPCertification/results/benchmark/${NETWORK_NAME}_${EPSILON}/${DATE}_BB-alpha-beta-CROWN_${NOM_RUN}"
mkdir -p "$LOG_DIR"

SUMMARY_FILE="$LOG_DIR/summary.csv"

echo "📊 Summary créé : $SUMMARY_FILE"


echo "start,end,verified_accuracy,total_examples,total_verified,total_falsified,timeout,mean_time,max_time" > "$SUMMARY_FILE"


echo "📁 Logs stockés dans : $LOG_DIR"

indexes_interval_cifar10=(
  "0 56" "58 62" "64 74" "77 77" "81 86" "89 89"
  "92 93" "95 95" "100 100" "103 104" "106 107"
  "111 111" "115 117" "128 129" "135 135"
  "139 139" "148 148" "155 155" "165 165"
)

indexes_interval_mnist=(
  "0 71" "73 76" "79 80" "82 86" "89 91" "94 94"
  "96 97" "100 100" "109 109" "117 117" "120 120"
  "122 122" "125 125" "132 132" "137 138" "145 145" 
  "173 173"
)

if [ "$DATASET_NAME" == "cifar10" ]; then
  indexes_interval=("${indexes_interval_cifar10[@]}")
else
  indexes_interval=("${indexes_interval_mnist[@]}")
fi


for interval in "${indexes_interval[@]}"; do
  read START END <<< "$interval"

  for ((i=START; i<=END; i++)); do
    RUN_START=$i
    RUN_END=$((i + 1))

    LOG_FILE="$LOG_DIR/run_${RUN_START}.log"

    {
      echo "=============================="
      echo "Run: index=$RUN_START"
      echo "Début: $(date)"
      echo "=============================="

      python abcrown.py \
        --config exp_configs/tutorial_examples/custom_margot_$NETWORK_NAME.yaml \
        --epsilon "$EPSILON" \
        --start "$RUN_START" \
        --end "$RUN_END"

      echo "=============================="
      echo "Fin: $(date)"
      echo "=============================="
    } > "$LOG_FILE" 2>&1

    echo "✅ Terminé: index=$RUN_START"

    # 🔎 Extraction des infos importantes → summary.csv
    awk -v s="$RUN_START" -v e="$RUN_END" '
    BEGIN {
      acc="None"; total="None"
      verified="None"; falsified="None"; timeout="None"
      mean="None"; max="None"
    }

    /Final verified acc:/ {
      if (match($0, /Final verified acc: ([0-9.]+)%/, m)) acc=m[1]
      if (match($0, /total ([0-9]+) examples/, m)) total=m[1]
    }

    /Problem instances count:/ {
      if (match($0, /total verified \(safe\/unsat\): ([0-9]+)/, m)) verified=m[1]
      if (match($0, /total falsified \(unsafe\/sat\): ([0-9]+)/, m)) falsified=m[1]
      if (match($0, /timeout: ([0-9]+)/, m)) timeout=m[1]
    }

    /mean time for ALL instances/ {
      if (match($0, /instances.*:([0-9.]+)/, m)) mean=m[1]
      if (match($0, /max time: ([0-9.]+)/, m)) max=m[1]
    }

    END {
      print s","e","acc","total","verified","falsified","timeout","mean","max
    }
    ' "$LOG_FILE" >> "$SUMMARY_FILE"

  done
done
