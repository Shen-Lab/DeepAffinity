#!/bin/bash
# POSIX

show_help() {
cat << EOF
Usage: ${0##*/} [-h] [-p|--prot PROTEIN_FILE] [-c|--comp COMPOUND_FILE] [-l|--label LABEL_FILE]

Run the DeepAffinity with given data or run our code with default IC50 dataset.

    no argument                    run with default IC50 dataset
    -h                             display this help and exit
    -o|--output OUTPUT_FILE_NAME   specify output filename
    -p|--prot PROTEIN_SPS_FILE     replace test_sps with user's protein sps format file
    -c|--comp COMPOUND_SMILE_FILE  replace test_smile with user's compound SIMLE format file
    -l|--label LABEL_FILE          replace test_ic50 with user's label IC50 file
    -t|--labeltype LABEL_TYPE      specify the label type you are using, it can be either IC50(default), EC50, Ki or Kd

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
    python joint-Model.py|tee output
    echo "Result has been saved to file ./Joint_models/joint_attention/joint_warm_start/output"
elif [ "$#" -eq 2 ]; then
    output=$2
    echo "dealing with data"
    cp data/dataset/IC50.tar.xz Joint_models/joint_attention/joint_warm_start/data/
    cd Joint_models/joint_attention/joint_warm_start/data/
    tar -xf IC50.tar.xz
    echo "Running default DeepAffinity"
    cd ..
    python joint-Model.py|tee $output
    echo "Result has been saved to file ./Joint_models/joint_attention/joint_warm_start/$output"
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
        -t|--labeltype)
            if [ "$2" ]; then
                labeltype=$2
                shift
            else
                die 'ERROR: "--labeltype" requires a non-empty option argument.'
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

if ! [ -v output ]; then
    output="./output"
fi

if ! [ -v labeltype ]; then
    labeltype="IC50"
fi

if ! [ "$labeltype" = "IC50" ] && ! [ "$labeltype" = "EC50" ] && ! [ "$labeltype" = "Ki" ] && ! [ "$labeltype" = "Kd" ]; then
    die  'ERROR: labeltype can only be "IC50", "EC50", "Ki" or "Kd".'
fi
num1=$(cat $prot|wc -l)
num2=$(cat $comp|wc -l)
if ! [ $num1 -eq $num2 ]; then
    die 'Error! Different Number of lines in $prot and $comp'
fi

if [ -v checkpoint ]; then 
    model_path="./Joint_models/joint_attention/testing/"
    file_name="joint-Model_test.py"
else
    model_path="./Joint_models/joint_attention/joint_warm_start/"
    file_name="joint-Model.py"
fi
echo "dealing with data"
tar -xf ./data/dataset/${labeltype}.tar.xz
cp $prot ./${labeltype}/SPS/test_sps
cp $comp ./${labeltype}/SPS/test_smile
cp $label ./${labeltype}/SPS/test_ic50
cp ./${labeltype}/SPS/* ${model_path}data/
rm -r ./${labeltype}
echo "Running default DeepAffinity"
cd ${model_path}
python ${file_name} $checkpoint $labeltype|tee $output
echo "Result has been saved to file ${model_path}${output}"
fi
