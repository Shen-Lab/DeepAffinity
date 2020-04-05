#!/bin/bash
# POSIX

show_help() {
cat << EOF
Usage: ${0##*/} [-h] [-p|--prot PROTEIN_FILE] [-c|--comp COMPOUND_FILE] [-l|--label LABEL_FILE]

Replace test data with user's data.
 
    -h          display this help and exit
    -p|--prot PROTEIN_FILE  replace test_sps with user's protein sps format file
    -c|--comp COMPOUND_FILE  replace test_smile with user's compound SIMLE format file
    -l|--label LABEL_FILE  replace test_ic50 with user's label IC50 file

EOF
}

die() {
     printf '%s\n' "$1" >&2
     exit 1
}

while :; do
    case $1 in
         -h|-\?|--help)
             show_help    # Display a usage synopsis.
             exit
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
                die 'ERROR: "--prot" requires a non-empty option argument.'
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

cp $prot "./data/test_sps"
cp $comp "./data/test_smile"
cp $label "./data/test_ic50"
