
#!/bin/bash
set -e

DATASET_NAME=$1
NETWORK_NAME=$2
NOM_RUN=$3
EPSILON=$4


if [ -z "$DATASET_NAME" ] || ([ "$DATASET_NAME" != "cifar10" ] && [ "$DATASET_NAME" != "mnist" ] && [ "$DATASET_NAME" != "cifar100" ]); then
  echo "Usage: $0 DATASET_NAME NETWORK_NAME NOM_RUN [EPSILON]"
  echo "DATASET_NAME must be 'cifar10', 'mnist', or 'cifar100'"
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

indexes_interval_cifar100=(
 "0 247" "249 366" "368 386" "388 447" "449 550" 
 "552 558" "560 562" "564 593" "595 598" "600 602" 
 "604 613" "616 626" "628 629" "631 643" "646 651" 
 "653 665" "667 671" "673 679" "681 681"  "683 684" 
 "686 688" "690 697" "699 711" "713 719" "721 721"  
 "723 723"  "725 734" "736 737" "739 739"  "741 749" 
 "751 753" "755 755"  "757 761" "763 765" "768 775" 
 "777 784" "786 786"  "788 791" "793 793"  "795 797" 
 "799 799"  "801 801"  "803 803"  "805 812" "815 817" 
 "819 821" "823 827" "830 831" "833 836" "838 838"  
 "840 843" "845 847" "850 852" "854 854"  "857 857"  
 "859 859"  "861 862" "865 869" "872 873" "875 876" 
 "879 882" "884 884"  "887 887"  "889 889"  "891 892" 
 "894 896" "898 898"  "902 902"  "904 906" "908 911"
  "914 914"  "916 916"  "918 918"  "921 921"  "925 925"  
  "927 928" "933 935" "937 939" "942 948" "950 953" 
  "956 957" "959 967" "969 969"  "971 971"  "974 974"  
  "977 977"  "980 981" "984 986" "991 992" "997 999" 
  "1001 1003" "1005 1006" "1009 1009"  "1012 1012"  
  "1014 1014"  "1016 1019" "1024 1026" "1030 1030"  
  "1032 1032"  "1037 1037"  "1039 1039"  "1041 1043" 
  "1045 1045"  "1047 1047"  "1056 1056"  "1061 1061"  
  "1064 1067" "1070 1070"  "1076 1076"  "1079 1079"  
  "1081 1081"  "1083 1083"  "1087 1087"  "1089 1089"  
  "1100 1100"  "1106 1106"  "1109 1110" "1112 1113" 
  "1116 1116"  "1119 1119"  "1121 1121"  "1125 1125"  
  "1128 1130" "1135 1135"  "1138 1138"  "1150 1151" 
  "1159 1159"  "1161 1161"  "1165 1165"  "1173 1173"  
  "1176 1176"  "1179 1179"  "1186 1186"  "1188 1189" 
  "1192 1193" "1195 1196" "1198 1198"  "1201 1201"  
  "1204 1204"  "1206 1206"  "1211 1211"  "1215 1215"  
  "1226 1227" "1244 1244"  "1252 1252"  "1257 1258" 
  "1269 1269"  "1276 1276"  "1280 1281" "1284 1284" 
  "1288 1289" "1292 1292"  "1297 1297"  "1300 1301"
  "1319 1320" "1327 1327"  "1340 1340"  "1344 1344" 
  "1347 1347"  "1354 1355" "1358 1358"  "1363 1363"
  "1369 1369"  "1375 1375"  "1382 1382"  "1407 1407"  
  "1414 1414"  "1433 1433"  "1442 1442"  "1452 1452"  
  "1471 1471"  "1482 1482"  "1519 1519"  "1532 1532"  
  "1544 1544"  "1549 1549"  "1620 1620"  "1628 1628"  
  "1640 1640"  "1679 1679"  "1743 1743"  "1753 1753"  "1872 1872"
)

if [ "$DATASET_NAME" == "cifar10" ]; then
  indexes_interval=("${indexes_interval_cifar10[@]}")
elif [ "$DATASET_NAME" == "cifar100" ]; then
  indexes_interval=("${indexes_interval_cifar100[@]}")
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
