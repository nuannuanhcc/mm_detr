export CONFIG_DIR="deform_detr"
export DIR_NAME="ddetr"

for i in $(seq 12 -1 0)
do
  let ITER=i
#  python3 ./tools/test.py ./am_configs/$CONFIG_DIR.py ./work_dirs/$DIR_NAME/epoch_$ITER.pth --work-dir './work_dirs/'$DIR_NAME''
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=295010 ./tools/dist_test.sh ./am_configs/$CONFIG_DIR.py ./work_dirs/$DIR_NAME/epoch_$ITER.pth 8 --work-dir './work_dirs/'$DIR_NAME''
#  CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=295011 ./tools/dist_test.sh ./am_configs/$CONFIG_DIR.py ./work_dirs/$DIR_NAME/epoch_$ITER.pth 4 --work-dir './work_dirs/'$DIR_NAME''
#  CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=295012 ./tools/dist_test.sh ./am_configs/$CONFIG_DIR.py ./work_dirs/$DIR_NAME/epoch_$ITER.pth 4 --work-dir './work_dirs/'$DIR_NAME''
done


