import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_all_episodes, save_to_hdf5


def parse_args():
    parser = argparse.ArgumentParser(description="Process raw episode logs into HDF5 dataset.")
    parser.add_argument('--log-dir',  type=Path, default=Path('data/raw'),       help="Root folder containing numbered episode directories.")
    parser.add_argument('--out',      type=Path, default=Path('data/processed/dataset.h5'), help="Output HDF5 file path.")
    parser.add_argument('--arm',      type=str,  default='arm_left',             help="Which arm log to load (arm_left or arm_right).")
    parser.add_argument('--min-steps',type=int,  default=50,                     help="Minimum timesteps for an episode to be included.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.log_dir.exists():
        print(f"[ERROR] Log directory not found: {args.log_dir}")
        sys.exit(1)

    print(f"Loading episodes from : {args.log_dir}")
    print(f"Arm                   : {args.arm}")
    print(f"Output                : {args.out}")
    print(f"Min steps             : {args.min_steps}")
    print()

    episodes = load_all_episodes(args.log_dir, arm=args.arm)

    before = len(episodes)
    episodes = [ep for ep in episodes if len(ep.ee_pos) >= args.min_steps]
    dropped  = before - len(episodes)
    if dropped:
        print(f"Dropped {dropped} episode(s) with fewer than {args.min_steps} steps.")

    if not episodes:
        print("[ERROR] No valid episodes after filtering.")
        sys.exit(1)

    lengths = [len(ep.ee_pos) for ep in episodes]
    modes   = [ep.mode for ep in episodes]
    print(f"\nEpisodes kept  : {len(episodes)}")
    print(f"Unimanual      : {modes.count(0)}")
    print(f"Bimanual       : {modes.count(1)}")
    print(f"Steps — min    : {min(lengths)}")
    print(f"Steps — max    : {max(lengths)}")
    print(f"Steps — mean   : {sum(lengths) / len(lengths):.0f}")
    print(f"Steps — total  : {sum(lengths)}")
    print()

    save_to_hdf5(episodes, args.out)


if __name__ == '__main__':
    main()