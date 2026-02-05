export function normalizeCommCollectPayload(form) {
  const toInt = (v, fallback) => {
    const n = Number(v);
    return Number.isFinite(n) ? Math.trunc(n) : fallback;
  };

  const uniqNonEmpty = (arr) =>
    (arr || [])
      .map((x) => String(x ?? "").trim())
      .filter(Boolean)
      .filter((v, i, a) => a.indexOf(v) === i);

  return {
    name: String(form.name ?? "").trim(),
    partition: String(form.partition ?? "").trim() || "normal",
    nodes: toInt(form.nodes, 1),
    ntasksPerNode: toInt(form.ntasksPerNode, 1),
    mem: String(form.mem ?? "").trim() || "100G",
    gres: String(form.gres ?? "").trim() || "dcu:4",
    exclusive: !!form.exclusive,
    noRequeue: !!form.noRequeue,
    timeLimit: String(form.timeLimit ?? "").trim() || "",

    appHome: String(form.appHome ?? "").trim(),
    appEntry: String(form.appEntry ?? "").trim() || "./app-run.sh",

    cleanupOnFail: !!form.cleanupOnFail,

    moduleStrategy: form.moduleStrategy === "custom" ? "custom" : "preset",
    presetModules: uniqNonEmpty(form.presetModules),
    customModules: uniqNonEmpty((form.customModules || []).map((m) => m?.moduleText)),
    moduleRemoves: uniqNonEmpty(form.moduleRemoves),

    envExports: (form.envExports || [])
      .map((e) => ({
        key: String(e?.key ?? "").trim(),
        value: String(e?.value ?? "").trim(),
      }))
      .filter((e) => e.key && e.value),

    mpirunArgs: (form.mpirunArgs || [])
      .map((a) => ({
        key: String(a?.key ?? "").trim(),
        value: String(a?.value ?? "").trim(),
      }))
      .filter((a) => a.key),
  };
}

function escapeDoubleQuotes(s) {
  return String(s ?? "").replace(/\\/g, "\\\\").replace(/\"/g, "\\\"");
}

export function generateSubScript(cfg) {
  const nodes = cfg.nodes;
  const ppn = cfg.ntasksPerNode;

  const moduleLines = [];
  moduleLines.push("module purge");

  const moduleLoads = cfg.moduleStrategy === "custom" ? cfg.customModules : cfg.presetModules;
  for (const m of moduleLoads || []) {
    moduleLines.push(`module load ${m}`);
  }
  for (const m of cfg.moduleRemoves || []) {
    moduleLines.push(`module rm ${m}`);
  }

  const exportLines = [];
  for (const e of cfg.envExports || []) {
    exportLines.push(`export ${e.key}="${escapeDoubleQuotes(e.value)}"`);
  }

  const ntasksExpr = "$( ( NNODE * PPN ) )".replace(/\s+/g, " ");
  const slurmExtra = cfg.timeLimit ? `#SBATCH --time=${cfg.timeLimit}\n` : "";

  const mpirunTokens = [];
  for (const a of cfg.mpirunArgs || []) {
    if (!a.key) continue;
    mpirunTokens.push(a.key);
    if (a.value) mpirunTokens.push(a.value);
  }

  const cmdArrayLines = [
    "CMD=(",
    '    "mpirun"',
    ...mpirunTokens.map((t) => `    "${escapeDoubleQuotes(t)}"`),
    `    "${escapeDoubleQuotes(cfg.appEntry)}"`,
    ")",
  ].join("\n");

  const cleanupBlock = cfg.cleanupOnFail
    ? "rm -rf ${INTERCEPT_LOG_PATH}"
    : 'echo "cleanupOnFail=false, keep logs at: ${INTERCEPT_LOG_PATH}"';

  return `#!/bin/bash
# slurm configurations
NNODE=${nodes}
PPN=${ppn}
NTASK=$(( NNODE * PPN ))
JOBNAME=${cfg.name}

WORK_HOME="$(pwd)"
APP_HOME="${escapeDoubleQuotes(cfg.appHome)}"

SLURM_LOG_PATH="${'${WORK_HOME}'}/slurm.logs"
if [ ! -d "${'${SLURM_LOG_PATH}'}" ]; then
    mkdir -p ${'${SLURM_LOG_PATH}'}
fi
INTERCEPT_LOG_HOME="${'${WORK_HOME}'}/logs"

sbatch << END
#!/bin/bash
#SBATCH --job-name=${'${JOBNAME}'}
#SBATCH --partition=${escapeDoubleQuotes(cfg.partition)}
#SBATCH --nodes=${'${NNODE}'}
#SBATCH --ntasks-per-node=${'${PPN}'}
#SBATCH --mem=${escapeDoubleQuotes(cfg.mem)}
#SBATCH --gres=${escapeDoubleQuotes(cfg.gres)}
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --output=${'${SLURM_LOG_PATH}'}/%j.out
#SBATCH --error=${'${SLURM_LOG_PATH}'}/%j.err
${slurmExtra}
job_id=\${SLURM_JOB_ID}

${moduleLines.join("\n")}

archive_name="\${SLURM_JOB_NAME}_${nodes}node_${nodes * ppn}proc_\${job_id}"
export INTERCEPT_LOG_PATH="${'${INTERCEPT_LOG_HOME}'}/\${archive_name}"
if [ ! -d "\${INTERCEPT_LOG_PATH}" ]; then
    mkdir -p \${INTERCEPT_LOG_PATH}
fi

${exportLines.join("\n")}

cd "${'${APP_HOME}'}"

${cmdArrayLines}

if ! "\${CMD[@]}"; then
  echo "Job failed, removing intercept log directory: \${INTERCEPT_LOG_PATH}"
  ${cleanupBlock}
fi
END
`;
}
