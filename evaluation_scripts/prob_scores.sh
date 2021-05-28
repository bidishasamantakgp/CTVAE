directory="${1}"
model_file="${2}"
sps="${3}"

this="${0}"
script_dir="$(dirname ${this})"

echo "directory: $directory"
echo "model_file: ${model_file}"
echo "samples per sentence: ${sps}"

mkdir -p ${directory}/prob

echo "getting prob score per sentence"

# # get prob values
# for x in {0..19}; do
#   echo -e "\n$x"
#   python ${script_dir}/prob_scores.py ${model_file} ${directory}/post.$x ${directory}/prob/post.$x
# done

# parallel execution | 5 parallel
for i in {0..19..5}; do
  for j in {0..4}; do
    x=$(($i+$j))
    echo -e "\n$x"
    python ${script_dir}/prob_scores.py ${model_file} ${directory}/post.$x ${directory}/prob/post.$x &
    done
  wait
done


# copy labels interpolated
# original labels per sentence, required for the next step
# ignores blank lines in the generated senence file
# and saves original label only for the non blank sentences
for x in {0..19}; do
  python ${script_dir}/copy_labels_interpolated.py ${directory}/labels1.txt ${directory}/post.$x $sps ${directory}/prob/post.$x.original_labels
done


echo "calculating score per file"

# analyse
for x in {0..19}; do
  echo -ne "\r$x "
  python ${script_dir}/analyse_probs_by_label.py ${directory}/prob/post.$x ${directory}/prob/post.$x.original_labels > ${directory}/prob/post.$x.analysis
  done
echo

echo "DONE"
