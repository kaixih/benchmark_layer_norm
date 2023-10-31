cd /home/tmp/benchmark_layer_norm
rm -rf nsys/*

file="shapes1.txt"
if [ ! -f "$file" ]; then
  echo "File not found"
  exit 1
fi

while IFS=' ' read -r hidden seqlen batch; do
  nsys profile -o nsys/result -f true python ln_fwd.py "$hidden" "$seqlen" "$batch" &> /dev/null
  nsys stats -r cuda_gpu_kern_sum --force-overwrite true --force-export true -o nsys/result nsys/result.nsys-rep &> /dev/null
  sum=0
  count=0
  tail -n +2 "nsys/result_cuda_gpu_kern_sum.csv" > nsys/result.csv
  while read line; do
    instances=$(echo $line | awk -F ',' '{print $3}')
    med=$(echo $line | awk -F ',' '{print $5}')
    if [ $instances -ge 100 ]; then
      sum=$(awk "BEGIN {printf \"%.2f\", $sum+$med}")
      count=$(($count+1))
    fi
  done < "nsys/result.csv"
  time=$(awk "BEGIN {printf \"%.2f\", $sum/1000}")
  echo "$hidden" "$seqlen" "$batch" "$time" ms "$count" fusions
done < "$file"


