#!/bin/bash
# POSIX

show_help() {
cat << EOF
Usage: ${0##*/} [-h] [-p|--prot PROTEIN_FILE] [-c|--comp COMPOUND_FILE] [-l|--label LABEL_FILE]

Run the DeepAffinity with given data or run our code with default IC50 dataset.

    no argument                    run with default IC50 dataset
    -h                             display this help and exit
    -o|--output OUTPUT_PATH        specify output path and filename
    -p|--prot PROTEIN_SPS_FILE     replace test_sps with user's protein sps format file
    -c|--comp COMPOUND_SMILE_FILE  replace test_smile with user's compound SIMLE format file
    -l|--label LABEL_FILE          replace test_ic50 with user's label IC50 file

EOF
}

die() {
     printf '%s\n' "$1" >&2
     exit 1
}

if [ "$#" -eq 0 ]; then
    echo "dealing with data"
    cp data/dataset/IC50.tar.xz Joint_models/joint_attention/joint_warm_start/data/
    cd Joint_models/joint_attention/joint_warm_start/data/
    tar -xf IC50.tar.xz
    echo "Running default DeepAffinity"
    cd ..
    python joint_Model.py|tee output
    echo "Result has been saved to file output"
elif [ "$#" -eq 2 ]; then
    output=$2
    echo "dealing with data"
    cp data/dataset/IC50.tar.xz Joint_models/joint_attention/joint_warm_start/data/
    cd Joint_models/joint_attention/joint_warm_start/data/
    tar -xf IC50.tar.xz
    echo "Running default DeepAffinity"
    cd ..
    python joint_Model.py|tee $output
    echo "Result has been saved to file $output"
else
while :; do
    case $1 in
        -h|-\?|--help)
            show_help    # Display a usage synopsis.
            exit
            ;;
        -m|--checkpoint)
            if [ "$2" ]; then
               checkpoint=$2
               shift
            else
                die 'ERROR: "--checkpoint" requires a non-empty option argument.'
            fi
            ;;
        -p|--prot)
            if [ "$2" ]; then
               prot=$2
               shift
            else
                die 'ERROR: "--prot" requires a non-empty option argument.'
            fi   
            ;;
        -c|--comp)
            if [ "$2" ]; then
               comp=$2
               shift
            else
                die 'ERROR: "--comp" requires a non-empty option argument.'
            fi
            ;;
        -l|--label)
            if [ "$2" ]; then
               label=$2
               shift
            else
                die 'ERROR: "--label" requires a non-empty option argument.'
            fi
            ;;
        -o|--output)
            if [ "$2" ]; then
               output=$2
               shift
            else
                die 'ERROR: "--output" requires a non-empty option argument.'
            fi
            ;;
        -?*)
             printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
             ;;
        *) 
            break
    esac
    shift
done

if ! [ -v output]; then
    output="./output"
fi

num1=$(cat $prot|wc -l)
num2=$(cat $comp|wc -l)
if ! [ num1 -eq num2]; then
    die 'Error! Different Number of lines in $prot and $comp'
fi

if [ -v checkpoint]; then 
    model_path="./Joint_models/joint_attention/testing/"
    file_name="joint_Model_test.py"
else
    model_path="./Joint_models/joint_attention/joint_warm_start/"
    file_name="joint_Model.py"
fi
echo "dealing with data"
tar -xf ./data/dataset/IC50.tar.xz
cp $prot ./data/dataset/IC50/SPS/test_sps
cp $comp ./data/dataset/IC50/SPS/test_smile
cp $label ./data/dataset/IC50/SPS/test_ic50
cp ./data/dataset/IC50/SPS/* ${model_path}data/
echo "Running default DeepAffinity"
python ${model_path}${file_name} $checkpoint|tee $output
echo "Result has been saved to file $output"
fi
