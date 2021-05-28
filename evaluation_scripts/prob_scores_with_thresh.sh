directory="${1}"
cosine_th="${2}"
overlap_th="${3}"

this="${0}"
script_dir="$(dirname ${this})"

echo "directory: $directory"
echo "cosine_th: $cosine_th"
echo "overlap_th: $overlap_th"

mkdir -p ${directory}/cosine

echo "calculating the scores after cosine and overlap threshold"

for i in {0..19..5}; do
  for j in {0..4}; do
    x=$(($i+$j))
    echo -e "\n$x"
    python ${script_dir}/analyse_probs_by_label_with_thresh.py ${directory}/prob/post.$x ${directory}/labels_all.txt ${directory}/original_sentences_all.txt ${directory}/post.$x ${cosine_th} ${overlap_th} > ${directory}/prob/post.$x.cosine_analysis &
  done
  wait
done

echo
echo "DONE"
