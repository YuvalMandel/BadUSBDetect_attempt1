import os
import glob
import random
import argparse

# --- CONFIGURATION ---
SOURCE_BASELINE = os.path.join("s2", "baseline")
SOURCE_ROTATION = os.path.join("s2", "rotation")

# ANSI color codes for terminal output
YELLOW = '\033[93m'
RESET = '\033[0m'
GREEN = '\033[92m'

def create_dir(output_dir):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Folder created: {output_dir}")

def process_category(file_list, category_name, target_file_count, target_lines, output_dir):
    """
    Randomly selects files, extracts the required number of lines, 
    and saves them to the output directory.
    """
    # Check if we have enough files
    actual_file_count = min(target_file_count, len(file_list))
    
    if target_file_count > len(file_list):
        print(f"{YELLOW}⚠ WARNING: Requested {target_file_count} files for '{category_name}', "
              f"but only found {len(file_list)}. Taking all available.{RESET}")

    # Randomly sample the required number of files
    selected_files = random.sample(file_list, actual_file_count)
    
    files_processed = 0
    short_files_count = 0

    for i, filepath in enumerate(selected_files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

        # Check if we have enough lines
        actual_lines = min(target_lines, len(lines))
        if target_lines > len(lines):
            short_files_count += 1

        # Create a new filename (e.g., Human_baseline_001.txt)
        new_filename = f"Human_{category_name}_{i+1:03d}.txt"
        out_filepath = os.path.join(output_dir, new_filename)

        # Write the exact number of lines
        with open(out_filepath, 'w', encoding='utf-8') as fout:
            fout.writelines(lines[:actual_lines])
            
        files_processed += 1

    # Print a single aggregate warning for short files to avoid terminal spam
    if short_files_count > 0:
        print(f"{YELLOW}⚠ WARNING: {short_files_count} files in '{category_name}' "
              f"had fewer than {target_lines} lines. Took maximum available.{RESET}")

    return files_processed

def main():
    parser = argparse.ArgumentParser(description="Balance and extract Human dataset from s2 folder.")
    
    parser.add_argument("-o", "--output_dir", type=str, required=True, 
                        help="Path to the output folder (e.g., Balanced_Humans)")
    parser.add_argument("-f", "--num_files", type=int, required=True, 
                        help="TOTAL number of files wanted (will be split 50/50)")
    parser.add_argument("-l", "--num_lines", type=int, required=True, 
                        help="Number of lines to extract per file")
    # NEW ARGUMENT HERE
    parser.add_argument("-e", "--ending", type=str, choices=['0', '1', 'all'], default='1',
                        help="Filter files by ending: '1' for *1.txt, '0' for *0.txt, or 'all' for *.txt (default: 1)")
    
    args = parser.parse_args()

    # 1. Validate directories
    if not os.path.exists(SOURCE_BASELINE) or not os.path.exists(SOURCE_ROTATION):
        print(f"❌ ERROR: Source folders '{SOURCE_BASELINE}' or '{SOURCE_ROTATION}' not found!")
        return

    create_dir(args.output_dir)

    # 2. Determine search pattern based on the user's choice
    if args.ending == 'all':
        search_pattern = "*.txt"
    else:
        search_pattern = f"*{args.ending}.txt"

    # Find all valid files matching the pattern
    # recursive=True allows searching in subdirectories if data is structured that way
    baseline_files = glob.glob(os.path.join(SOURCE_BASELINE, "**", search_pattern), recursive=True)
    rotation_files = glob.glob(os.path.join(SOURCE_ROTATION, "**", search_pattern), recursive=True)

    print("🔍 Scanning source directories...")
    print(f"Filter applied: {search_pattern}")
    print(f"Found {len(baseline_files)} files in Baseline.")
    print(f"Found {len(rotation_files)} files in Rotation.\n")

    if len(baseline_files) == 0 and len(rotation_files) == 0:
        print(f"❌ ERROR: No files matching '{search_pattern}' were found in the source directories.")
        return

    # 3. Calculate 50/50 split
    target_per_category = args.num_files // 2
    
    # If the user asked for an odd number (e.g., 21), warn them that it will be rounded down to even
    if args.num_files % 2 != 0:
        print(f"{YELLOW}⚠ INFO: Requested an odd number of files ({args.num_files}). "
              f"Rounding to {target_per_category * 2} for a perfect 50/50 split.{RESET}\n")

    print(f"🚀 Processing Baseline data (Target: {target_per_category} files)...")
    base_processed = process_category(baseline_files, "baseline", target_per_category, args.num_lines, args.output_dir)

    print(f"\n🚀 Processing Rotation data (Target: {target_per_category} files)...")
    rot_processed = process_category(rotation_files, "rotation", target_per_category, args.num_lines, args.output_dir)

    total = base_processed + rot_processed
    
    # 4. Final Summary
    print("-" * 50)
    print(f"{GREEN}✅ Done! Successfully created a balanced human dataset.{RESET}")
    print(f"Total files saved: {total} ({base_processed} Baseline, {rot_processed} Rotation)")
    print(f"Lines per file: {args.num_lines} (or max available)")
    print(f"Output Directory: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
