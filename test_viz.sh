#!/bin/bash

set -e
set -o pipefail
set -u

ACC_DIR="data/smartfallmm/young/accelerometer/watch"
GYRO_DIR="data/smartfallmm/young/gyroscope/watch"
SKL_DIR="data/smartfallmm/young/skeleton"
FS=30.0
FILTER_TYPE="madgwick"
DEVICE="0 1"
NUM_WORKER=48
MAIN_OUT="viz_young_all"
WRIST_JOINT_IDX=9
IS_LINEAR_ACC=true

mkdir -p "${MAIN_OUT}"

echo "================================================================"
echo " SLURM Job:      $SLURM_JOB_ID"
echo " Hostname:       $(hostname)"
echo " ACC_DIR:        ${ACC_DIR}"
echo " GYRO_DIR:       ${GYRO_DIR}"
echo " SKL_DIR:        ${SKL_DIR}"
echo " Filter Type:    ${FILTER_TYPE}"
echo " Resample Rate:  ${FS}"
echo " Wrist Joint:    ${WRIST_JOINT_IDX}"
echo " Linear Acc:     ${IS_LINEAR_ACC}"
echo " GPUs:           ${DEVICE}"
echo " CPU workers:    ${NUM_WORKER}"
echo " Main Out:       ${MAIN_OUT}"
echo "================================================================"

[ ! -d "${ACC_DIR}" ] && { echo "ERROR: ACC_DIR not found: ${ACC_DIR}"; exit 1; }
[ ! -d "${GYRO_DIR}" ] && { echo "ERROR: GYRO_DIR not found: ${GYRO_DIR}"; exit 1; }
[ ! -d "${SKL_DIR}" ] && { echo "ERROR: SKL_DIR not found: ${SKL_DIR}"; exit 1; }
[ ! -f "visualize_skl_alignment.py" ] && { echo "ERROR: visualize_skl_alignment.py not found!"; exit 1; }

COUNT=0
PROCESSED=0
for ACC_FILE in "${ACC_DIR}"/*.csv; do
  FNAME=$(basename "${ACC_FILE}")
  GYR_FILE="${GYRO_DIR}/${FNAME}"
  SKL_FILE="${SKL_DIR}/${FNAME}"
  
  COUNT=$((COUNT+1))
  
  # Try to process even if some files are missing - let the Python script handle errors
  echo "=== [${COUNT}] Attempting to visualize ${FNAME} ==="
  echo "ACC:  ${ACC_FILE}"
  echo "GYRO: ${GYR_FILE}"
  echo "SKL:  ${SKL_FILE}"
  
  # Create output directory regardless
  OUT_SUBDIR="${MAIN_OUT}/${FNAME%.csv}"
  mkdir -p "${OUT_SUBDIR}"
  echo "Out:  ${OUT_SUBDIR}"
  
  LINEAR_FLAG=""
  if [ "${IS_LINEAR_ACC}" = "true" ]; then
    LINEAR_FLAG="--is-linear-acc"
  fi
 # Run the script - it will skip if files are missing
if python visualize_skl_alignment.py \
      --acc-file "${ACC_FILE}" \
      --gyro-file "${GYR_FILE}" \
      --skl-file "${SKL_FILE}" \
      --fs "${FS}" \
      --filter-type "${FILTER_TYPE}" \
      --out-dir "${OUT_SUBDIR}" \
      --wrist-joint-idx "${WRIST_JOINT_IDX}" \
      --assume-skl-positions \
      ${LINEAR_FLAG}; then
  PROCESSED=$((PROCESSED+1))
  echo "Successfully processed ${FNAME}. Check ${OUT_SUBDIR} for plots."
else
  echo "Skipped ${FNAME} due to missing files or processing errors."
  # Optionally remove the empty directory if no processing occurred
  rmdir "${OUT_SUBDIR}" 2>/dev/null || true
fi 
 echo "-----------------------------------------"
done

echo "Attempted processing of ${COUNT} files. Successfully processed ${PROCESSED} files."
echo "Results saved in ${MAIN_OUT}."
echo "Done."
