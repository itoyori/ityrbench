#!/bin/bash
[[ -z "${PS1+x}" ]] && set -euo pipefail

MPIEXEC=${MPIEXEC:-mpiexec}

$MPIEXEC --version || true

STDOUT_FILE=mpirun_out.txt

case $KOCHI_MACHINE in
  wisteria-o)
    export UTOFU_SWAP_PROTECT=1

    ityr_mpirun() {
      local n_processes=$1
      local n_processes_per_node=$2
      (
        vcoordfile=$(mktemp)
        if [[ $PJM_ENVIRONMENT == INTERACT ]]; then
          tee_cmd="tee $STDOUT_FILE"
          of_opt=""
          trap "rm -f $vcoordfile" EXIT
        else
          export PLE_MPI_STD_EMPTYFILE=off # do not create empty stdout/err files
          tee_cmd="cat"
          of_opt="-of-proc $STDOUT_FILE"
          trap "rm -f $vcoordfile; compgen -G ${STDOUT_FILE}.* && tail -n +1 \$(ls ${STDOUT_FILE}.* -v) | tee $STDOUT_FILE && rm ${STDOUT_FILE}.*" EXIT
          # trap "rm -f $vcoordfile; compgen -G ${STDOUT_FILE}.* && tail -n +1 \$(ls ${STDOUT_FILE}.* -v) > $STDOUT_FILE && head -100 $STDOUT_FILE" EXIT
        fi
        np=0
        if [[ -z ${PJM_NODE_Y+x} ]]; then
          # 1D
          for x in $(seq 1 $PJM_NODE_X); do
            for i in $(seq 1 $n_processes_per_node); do
              echo "($((x-1)))" >> $vcoordfile
              if (( ++np >= n_processes )); then
                break
              fi
            done
          done
        elif [[ -z ${PJM_NODE_Z+x} ]]; then
          # 2D
          for x in $(seq 1 $PJM_NODE_X); do
            for y in $(seq 1 $PJM_NODE_Y); do
              for i in $(seq 1 $n_processes_per_node); do
                echo "($((x-1)),$((y-1)))" >> $vcoordfile
                if (( ++np >= n_processes )); then
                  break 2
                fi
              done
            done
          done
        else
          # 3D
          for x in $(seq 1 $PJM_NODE_X); do
            for y in $(seq 1 $PJM_NODE_Y); do
              for z in $(seq 1 $PJM_NODE_Z); do
                for i in $(seq 1 $n_processes_per_node); do
                  echo "($((x-1)),$((y-1)),$((z-1)))" >> $vcoordfile
                  if (( ++np >= n_processes )); then
                    break 3
                  fi
                done
              done
            done
          done
        fi
        $MPIEXEC $of_opt -n $n_processes \
          --vcoordfile $vcoordfile \
          -- setarch $(uname -m) --addr-no-randomize "${@:3}" | $tee_cmd
      )
    }
    ;;
  squid-c)
    export UCX_TLS=rc_x,self,sm
    export OMPI_MCA_mca_base_env_list="TERM;LD_PRELOAD;UCX_NET_DEVICES;UCX_MAX_NUM_EPS=inf;UCX_TLS;"
    # export OMPI_MCA_mca_base_env_list="TERM;LD_PRELOAD;UCX_NET_DEVICES;UCX_MAX_NUM_EPS=inf;UCX_TLS;UCX_LOG_LEVEL=info;"
    # export OMPI_MCA_mca_base_env_list="TERM;LD_PRELOAD;UCX_NET_DEVICES;UCX_MAX_NUM_EPS=inf;UCX_TLS;UCX_LOG_LEVEL=func;UCX_LOG_FILE=ucxlog.%h.%p;"
    ityr_mpirun() {
      local n_processes=$1
      local n_processes_per_node=$2

      trap "compgen -G ${STDOUT_FILE}.* && tail -n +1 \$(ls ${STDOUT_FILE}.* -v) > $STDOUT_FILE && rm ${STDOUT_FILE}.*" EXIT

      $MPIEXEC -n $n_processes -N $n_processes_per_node \
        --output file=$STDOUT_FILE \
        --prtemca ras simulator \
        --prtemca plm_ssh_agent ssh \
        --prtemca plm_ssh_args " -i /sqfs/home/v60680/sshd/ssh_client_rsa_key -o StrictHostKeyChecking=no -p 50000 -q" \
        --hostfile $NQSII_MPINODES \
        --mca btl ^ofi \
        --mca osc_ucx_acc_single_intrinsic true \
        setarch $(uname -m) --addr-no-randomize "${@:3}"
    }
    ;;
  *)
    ityr_mpirun() {
      local n_processes=$1
      local n_processes_per_node=$2
      $MPIEXEC -n $n_processes -N $n_processes_per_node \
        --mca osc ucx \
        setarch $(uname -m) --addr-no-randomize "${@:3}"
    }
    ;;
esac

run_trace_viewer() {
  if [[ -z ${KOCHI_FORWARD_PORT+x} ]]; then
    echo "Trace viewer cannot be launched without 'kochi interact' command."
    exit 1
  fi
  shopt -s nullglob
  MLOG_VIEWER_ONESHOT=false bokeh serve $KOCHI_INSTALL_PREFIX_MASSIVELOGGER/viewer --port $KOCHI_FORWARD_PORT --allow-websocket-origin \* --session-token-expiration 3600 --args ityr_log_*.ignore
}

export ITYR_ENABLE_SHARED_MEMORY=$KOCHI_PARAM_SHARED_MEM
export ITYR_ORI_CACHE_SIZE=$(bc <<< "$KOCHI_PARAM_CACHE_SIZE * 2^20 / 1")
export ITYR_ORI_SUB_BLOCK_SIZE=$KOCHI_PARAM_SUB_BLOCK_SIZE
export ITYR_ORI_MAX_DIRTY_CACHE_SIZE=$(bc <<< "$KOCHI_PARAM_MAX_DIRTY * 2^20 / 1")

if [[ $KOCHI_PARAM_DEBUGGER == 1 ]] && [[ -z "${PS1+x}" ]]; then
  echo "Use kochi interact to run debugger."
  exit 1
fi
