#!/bin/bash --login

########## SBATCH Lines for Resource Request ##########

#SBATCH --time=119:59:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --gres=gpu:1                # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --mem=28G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name CNN_zenith_multiplefiles      # you can give your job a name for easier identification (same as -J)
#SBATCH --output /mnt/home/micall12/DNN_LE/make_jobs/run_CNN/CNN_multiplefiles_IC19_lr_contained_oldDC.out

########## Command Lines to Run ##########

INPUT="NuMu_140000_level2_uncleaned_cleanedpulsesonly_vertexDC_IC19_flat_95bins_36034evtperbin_file0?_CC.lt100.transformedinputstatic_transformed3output_contained.hdf5"
INDIR="/mnt/research/IceCube/jmicallef/DNN_files/"
OUTDIR="/mnt/home/micall12/LowEnergyNeuralNetwork"
NUMVAR=1
LR_EPOCH=50
LR_DROP=0.1
LR=0.001
OUTNAME="test"

START=0
END=14 #511
STEP=7
for ((EPOCH=$START;EPOCH<=$END;EPOCH+=$STEP));
do
    MODELNAME="$OUTDIR/output_plots/${OUTNAME}/${OUTNAME}_${EPOCH}epochs_model.hdf5"
    
    case $EPOCH in
    0)
        singularity exec -B /mnt/home/micall12:/mnt/home/micall12 -B /mnt/research/IceCube:/mnt/research/IceCube --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif python ${OUTDIR}/CNN_LoadMultipleFiles.py --input_files $INPUT --path $INDIR --output_dir $OUTDIR --name $OUTNAME -e $STEP --start $EPOCH --variables $NUMVAR --no_test True --first_variable zenith --lr $LR --lr_epoch $LR_EPOCH --lr_drop $LR_DROP
    ;;
    $END)
        singularity exec -B /mnt/home/micall12:/mnt/home/micall12 -B /mnt/research/IceCube:/mnt/research/IceCube --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif python ${OUTDIR}/CNN_LoadMultipleFiles.py --input_files $INPUT --path $INDIR --output_dir $OUTDIR --name $OUTNAME -e $STEP --start $EPOCH --variables $NUMVAR --model $MODELNAME --no_test False --first_variable zenith --lr $LR --lr_epoch $LR_EPOCH --lr_drop $LR_DROP
    ;;
    *)
        singularity exec -B /mnt/home/micall12:/mnt/home/micall12 -B /mnt/research/IceCube:/mnt/research/IceCube --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif python ${OUTDIR}/CNN_LoadMultipleFiles.py --input_files $INPUT --path $INDIR --output_dir $OUTDIR --name $OUTNAME -e $STEP --start $EPOCH --variables $NUMVAR --model $MODELNAME --no_test True --first_variable zenith --lr $LR --lr_epoch $LR_EPOCH --lr_drop $LR_DROP
    ;;
    esac
done
