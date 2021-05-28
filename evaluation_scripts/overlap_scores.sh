directory="${1}"
sps="${2}"

this="${0}"
script_dir="$(dirname ${this})"

echo "directory: $directory"
echo "samples per sentence: ${sps}"

mkdir -p ${directory}/prob

# replicate labels
python ${script_dir}/copy_labels.py ${directory}/labels1.txt $sps ${directory}/labels_all.txt

# replicate original sentences
python ${script_dir}/copy_labels.py ${directory}/original_sentence1 $sps ${directory}/original_sentences_all.txt

# overlap score # for 10 samples per sentence
# we take max for every score across all samples of a sentence,
#     then report mean and std

echo "calculating overlap score"

for x in {0..19}; do
  echo -ne "\r$x/20 "
  python ${script_dir}/overlap_by_labels.py ${directory}/original_sentences_all.txt ${directory}/labels_all.txt ${directory}/post.$x ${directory}/prob/post.$x $sps > prob/post.$x.overlap_scores
done
echo

echo "DONE"
