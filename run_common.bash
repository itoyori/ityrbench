#!/bin/bash
[[ -z "${PS1+x}" ]] && set -euo pipefail

MPIEXEC=${MPIEXEC:-mpiexec}

[[ -x "$(command -v $MPIEXEC)" ]] && $MPIEXEC --version || true

STDOUT_FILE=mpirun_out.txt

case $KOCHI_MACHINE in
  wisteria-o)
    export UTOFU_SWAP_PROTECT=1

    ityr_mpirun() {
      local n_processes=$1
      local n_processes_per_node=$2
      local bind_to=$3

      if [[ $bind_to == none ]]; then
        set_cpu_affinity=0
      else
        set_cpu_affinity=1
      fi

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
          --mca plm_ple_cpu_affinity $set_cpu_affinity \
          --mca plm_ple_numanode_assign_policy share_band \
          -- setarch $(uname -m) --addr-no-randomize "${@:4}" | $tee_cmd
      )
    }
    ;;
  squid-c)
    export OMPI_MCA_mca_base_env_list="TERM;UCX_NET_DEVICES;UCX_MAX_NUM_EPS=inf;"
    # export OMPI_MCA_mca_base_env_list="TERM;UCX_NET_DEVICES;UCX_MAX_NUM_EPS=inf;UCX_LOG_LEVEL=info;"
    # export OMPI_MCA_mca_base_env_list="TERM;UCX_NET_DEVICES;UCX_MAX_NUM_EPS=inf;UCX_LOG_LEVEL=func;UCX_LOG_FILE=ucxlog.%h.%p;"
    # export OMPI_MCA_mca_base_env_list="TERM;UCX_NET_DEVICES;UCX_MAX_NUM_EPS=inf;UCX_LOG_LEVEL=func;UCX_LOG_FILE=/dev/null;"
    ityr_mpirun() {
      local n_processes=$1
      local n_processes_per_node=$2
      local bind_to=$3

      (
        trap "compgen -G ${STDOUT_FILE}.* && tail -n +1 \$(ls ${STDOUT_FILE}.* -v) > $STDOUT_FILE && rm ${STDOUT_FILE}.*" EXIT

        # Workaround for the issue: https://github.com/openpmix/openpmix/issues/2980
        if [[ $MPIEXEC == mpitx ]]; then
          double_hyphen=--
        else
          double_hyphen=
        fi

        # About hcoll: https://github.com/open-mpi/ompi/issues/9885
        $MPIEXEC -n $n_processes -N $n_processes_per_node \
          --bind-to $bind_to \
          --output file=$STDOUT_FILE \
          --prtemca ras simulator \
          --prtemca plm_ssh_agent ssh \
          --prtemca plm_ssh_args " -i /sqfs/home/v60680/sshd/ssh_client_rsa_key -o StrictHostKeyChecking=no -p 50000 -q" \
          --prtemca plm_ssh_no_tree_spawn "true" \
          --hostfile $NQSII_MPINODES \
          --mca btl ^ofi \
          --mca osc_ucx_acc_single_intrinsic true \
          --mca coll_hcoll_enable 0 \
          $double_hyphen setarch $(uname -m) --addr-no-randomize "${@:4}"
      )
    }
    ;;
  *)
    # export OMPI_MCA_mca_base_env_list="UCX_LOG_LEVEL=func;UCX_LOG_FILE=ucxlog.%h.%p;"
    ityr_mpirun() {
      local n_processes=$1
      local n_processes_per_node=$2
      local bind_to=$3

      # Workaround for the issue: https://github.com/openpmix/openpmix/issues/2980
      if [[ $MPIEXEC == mpitx ]]; then
        double_hyphen=--
      else
        double_hyphen=
      fi

      $MPIEXEC -n $n_processes -N $n_processes_per_node \
        --bind-to $bind_to \
        --mca osc ucx \
        --mca pml_ucx_tls any \
        --mca pml_ucx_devices any \
        $double_hyphen setarch $(uname -m) --addr-no-randomize "${@:4}" | tee $STDOUT_FILE
    }
    ;;
esac

run_trace_viewer() {
  if [[ -z ${KOCHI_FORWARD_PORT+x} ]]; then
    echo "Trace viewer cannot be launched without 'kochi interact' command."
    exit 1
  fi
  shopt -s nullglob
  MLOG_VIEWER_ONESHOT=false bokeh serve ./third-party/massivelogger/viewer --port $KOCHI_FORWARD_PORT --allow-websocket-origin \* --session-token-expiration 3600 --args ityr_log_*.ignore
}

if [[ ! -z ${KOCHI_INSTALL_PREFIX_ITOYORI+x} ]]; then
  export ITYR_ENABLE_SHARED_MEMORY=$KOCHI_PARAM_SHARED_MEM
  export ITYR_ORI_CACHE_SIZE=$(bc <<< "$KOCHI_PARAM_CACHE_SIZE * 2^20 / 1")
  export ITYR_ORI_SUB_BLOCK_SIZE=$KOCHI_PARAM_SUB_BLOCK_SIZE
  export ITYR_ORI_MAX_DIRTY_CACHE_SIZE=$(bc <<< "$KOCHI_PARAM_MAX_DIRTY * 2^20 / 1")
  export ITYR_ORI_NONCOLL_ALLOCATOR_SIZE=$(bc <<< "$KOCHI_PARAM_NONCOLL_ALLOC_SIZE * 2^20 / 1")

  if [[ $KOCHI_PARAM_NODES == 1 ]] && [[ $ITYR_ENABLE_SHARED_MEMORY == 1 ]]; then
    export ITYR_ORI_CACHE_SIZE=65536
  fi
fi

if [[ ${KOCHI_PARAM_DEBUGGER:-0} == 1 ]] && [[ -z "${PS1+x}" ]]; then
  echo "Use kochi interact to run debugger."
  exit 1
fi
