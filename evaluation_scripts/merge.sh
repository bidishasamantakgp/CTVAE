directory="${1}"

this="${0}"
script_dir="$(dirname ${this})"

echo "directory: $directory"

mkdir -p ${directory}/merged

python ${script_dir}/merge.py ${directory}/post.{1..2} ${directory}/merged/post.0
python ${script_dir}/merge.py ${directory}/post.{3..4} ${directory}/merged/post.1
python ${script_dir}/merge.py ${directory}/post.{5..6} ${directory}/merged/post.2
python ${script_dir}/merge.py ${directory}/post.{7..8} ${directory}/merged/post.3
python ${script_dir}/merge.py ${directory}/post.{9..10} ${directory}/merged/post.4
python ${script_dir}/merge.py ${directory}/post.{11..12} ${directory}/merged/post.5
python ${script_dir}/merge.py ${directory}/post.{13..14} ${directory}/merged/post.6
python ${script_dir}/merge.py ${directory}/post.{15..16} ${directory}/merged/post.7
python ${script_dir}/merge.py ${directory}/post.{17..18} ${directory}/merged/post.8
python ${script_dir}/merge.py ${directory}/post.{19..20} ${directory}/merged/post.9

python ${script_dir}/merge.py ${directory}/original_sentences_all.txt ${directory}/original_sentences_all.txt ${directory}/merged/original_sentences_all.txt
python ${script_dir}/merge.py ${directory}/labels1.txt ${directory}/labels1.txt ${directory}/merged/labels1.txt

echo "DONE"
