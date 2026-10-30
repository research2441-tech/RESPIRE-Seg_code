"""Write the experiment queue (jobs.txt) in priority order."""
R = "/home/claude/runs"
B = "/home/claude/data/banks"
EVAL = ["acdc_cal", "acdc_test", "cmrx_real", "acdc_test_stress"]
EXT = ["mnms2_cal", "mnms2_test"]


def pred(tag, banks):
    return " && ".join(f"python3 predict.py --tag {tag} --bank {b}" for b in banks)


def tj(tag, args, evals=EVAL):
    return f"{R}/{tag}/vol_{evals[-1]}.json | python3 train.py {args} --tag {tag} && {pred(tag, evals)}"


def pj(tag, banks):
    return f"{R}/{tag}/vol_{banks[-1]}.json | {pred(tag, banks)}"


def bank(name):
    return f"{B}/{name}.npz | python3 build_bank.py {name}"


FULL = "--model respire --paired 0"
BASE = {"unet": "--model unet", "attunet": "--model attunet", "sharpunet": "--model sharpunet",
        "nnunet2d": "--model nnunet2d", "transunet": "--model transunet", "multitask": "--model multitask"}
MAIN = ["respire"] + list(BASE)
jobs = [bank("mnms2_test"), bank("mnms2_cal")]
# RESPIRE-Seg (all SDF-head models retrained after replacing the saturating tanh output by a linear head)
for sd in (0, 1):
    jobs.append(tj(f"respire_s{sd}", FULL + f" --seed {sd}"))
for t in ("unet_s0", "attunet_s0"):
    jobs.append(pj(t, EVAL))
for k in ("sharpunet", "nnunet2d", "transunet", "multitask"):
    jobs.append(tj(f"{k}_s0", BASE[k] + " --seed 0"))
for bk in ("clean", "imgwarp", "generic"):
    jobs.append(tj(f"unet{bk}_s0", f"--model unet --bank acdc_train_{bk} --seed 0"))
for k in ("unet", "attunet", "sharpunet"):   # second seed for the main comparators (compute budget)
    jobs.append(tj(f"{k}_s1", BASE[k] + " --seed 1"))
ABL = {"noA": "--model respire_noA --paired 0", "noG": "--model respire_noG --paired 0",
       "noB": "--model respire_noB --paired 0", "base": "--model respire_base --paired 0"}
for k, v in ABL.items():
    jobs.append(tj(f"abl{k}_s0", v + " --seed 0"))
for bk in ("notp", "1coil", "sine"):
    m, cmd = tj(f"unet{bk}_s0", f"--model unet --bank acdc_train_{bk} --seed 0").split(" | ", 1)
    jobs.append(f"{m} | ( [ -f {B}/acdc_train_{bk}.npz ] || python3 build_bank.py acdc_train_{bk} ) && {cmd}")
for t in ["respire_s0", "respire_s1", "unet_s0", "unet_s1", "attunet_s0", "attunet_s1", "sharpunet_s0", "sharpunet_s1",
          "nnunet2d_s0", "transunet_s0", "multitask_s0"]:
    jobs.append(pj(t, EXT))
jobs.append(tj("ablplusD_s0", "--model respire --paired 1 --seed 0"))
open("jobs.txt", "w").write("\n".join(jobs) + "\n")
print(len(jobs), "jobs")
