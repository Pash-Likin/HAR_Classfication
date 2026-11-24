"""复算full/PCA新协议结果并检查冻结选择；旧ANOVA快照需使用原版代码。"""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="官方数据ZIP或目录；默认项目data/")
    args = parser.parse_args()
    if (args.run / "search_plan.json").is_file():
        from run_experiment import verify_extended
        from har_study.data import prepare_dataset
        result = verify_extended(args.run, prepare_dataset(args.data, ROOT / "data"))
    else:
        from har_study.verify import verify_run
        result = verify_run(args.run)
    print(result)
    if (args.run / "stage_timing_environment.json").is_file():
        from har_study.stage_timing import verify_stage_timing
        print(verify_stage_timing(args.run))
