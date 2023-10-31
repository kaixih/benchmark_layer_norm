export WORKSPACE_DIR=/home
export ENABLE_CUDNN_LAYER_NORM=${1:-false}
cd $WORKSPACE_DIR/benchmark_layer_norm
mkdir -p nsys/
mkdir -p ln_fwd_dump/
rm -rf nsys/*
rm -rf ln_fwd_dump/*

file="shapes.txt"
if [ ! -f "$file" ]; then
  echo "File not found"
  exit 1
fi

export TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_MIN_VLOG_LEVEL=0 TF_CPP_VMODULE=cudnn_norm_rewriter=4

while IFS=' ' read -r hidden seqlen batch; do
  nsys profile -o nsys/result -f true python ln_fwd.py "$hidden" "$seqlen" "$batch" &> /dev/null
  nsys stats -r cuda_gpu_kern_sum --force-overwrite true --force-export true -o nsys/result nsys/result.nsys-rep &> /dev/null
  sum=0
  count=0
  tail -n +2 "nsys/result_cuda_gpu_kern_sum.csv" > nsys/tmp.csv
  while read line; do
    instances=$(echo $line | awk -F ',' '{print $3}')
    med=$(echo $line | awk -F ',' '{print $5}')
    if [ $instances -ge 100 ]; then
      sum=$(awk "BEGIN {printf \"%.2f\", $sum+$med}")
      count=$(($count+1))
    fi
  done < "nsys/tmp.csv"
  time=$(awk "BEGIN {printf \"%.2f\", $sum/1000}")
  echo "$hidden" "$seqlen" "$batch" "$time" ms "$count" fusions
done < "$file"



