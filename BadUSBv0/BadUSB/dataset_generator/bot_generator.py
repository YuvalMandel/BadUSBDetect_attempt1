import os
import random
import numpy as np
import argparse

# Base timestamp for start (in milliseconds, simulating Unix Epoch)
BASE_TIMESTAMP_MS = 1700000000000 

# Possible keys for input simulation
KEYS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["Space", "Enter", "Shift", "Alt"]

def create_dir(output_dir):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Folder created: {output_dir}")

def write_bot_log(filename, dwells, flights, output_dir):
    """Converts Dwell and Flight arrays into a text log in s2 format."""
    filepath = os.path.join(output_dir, filename)
    
    # Random start time shift to make files unique
    current_time = BASE_TIMESTAMP_MS + random.randint(0, 1000000)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for i in range(len(dwells)):
            key = random.choice(KEYS)
            
            # Flight Time (Pause BEFORE next key press)
            # For the first press, flight time doesn't make sense, just add a small pause
            if i > 0:
                current_time += int(flights[i])
            else:
                current_time += random.randint(10, 50)
                
            # Write KeyDown
            f.write(f"{key} KeyDown {current_time}\n")
            
            # Dwell Time (Key hold duration)
            current_time += int(dwells[i])
            
            # Write KeyUp
            f.write(f"{key} KeyUp {current_time}\n")

# ==============================================================================
# ATTACK GENERATORS (Unique parameters for each call)
# ==============================================================================

def generate_machine_gun(num_events):
    """Very fast, almost no noise."""
    base_val = random.randint(4, 12) # Each machine gun has its own base speed
    dwells = np.clip(np.random.normal(base_val, 0.5, num_events), 1, None)
    flights = np.clip(np.random.normal(base_val, 0.5, num_events), 1, None)
    return dwells, flights

def generate_the_robot(num_events):
    """Perfectly uniform, zero variance."""
    val = random.randint(1, 80)
    dwells = np.full(num_events, val)
    flights = np.full(num_events, val)
    return dwells, flights

def generate_gaussian_faker(num_events):
    """Tries to imitate a human with normal distribution."""
    mean_d = random.randint(70, 130)
    std_d = random.randint(10, 50)
    mean_f = random.randint(80, 150)
    std_f = random.randint(15, 60)
    
    dwells = np.clip(np.random.normal(mean_d, std_d, num_events), 10, None)
    flights = np.clip(np.random.normal(mean_f, std_f, num_events), 10, None)
    return dwells, flights

def generate_uniform_jitter(num_events):
    """Random chaos within defined boundaries."""
    low_d, high_d = random.randint(5, 55), random.randint(60, 150)
    low_f, high_f = random.randint(5, 55), random.randint(60, 200)
    
    dwells = np.random.uniform(low_d, high_d, num_events)
    flights = np.random.uniform(low_f, high_f, num_events)
    return dwells, flights

def generate_burst_mode(num_events):
    """Batch input: fast typing burst, pause, fast typing burst again."""
    dwells = []
    flights = []

    while len(dwells) < num_events:
        burst_size = random.randint(10, 30) # Length of "queue"
        pause_time = random.randint(200, 1000) # Pause between queues

        for _ in range(burst_size):
            dwells.append(random.randint(5, 15))
            flights.append(random.randint(5, 15))

        # Replace the last flight time with a long pause
        if len(flights) > 0:
            flights[-1] = pause_time

    # Trim to required length
    return np.array(dwells[:num_events]), np.array(flights[:num_events])


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    # Setup argparse for command line arguments
    parser = argparse.ArgumentParser(description="Generate synthetic BadUSB keystroke datasets.")
    
    parser.add_argument("-o", "--output_dir", type=str, default="Synthetic_Bots", 
                        help="Directory to save the generated files (default: Synthetic_Bots)")
    parser.add_argument("-f", "--files_per_type", type=int, default=100, 
                        help="Number of files to generate per attack type (default: 100)")
    parser.add_argument("-e", "--events_per_file", type=int, default=200,
                        help="Number of key press events per file (default: 200)")
    
    args = parser.parse_args()

    # Pass the output directory to the creation function
    create_dir(args.output_dir)
    
    attack_types = {
        "Machine_Gun":  generate_machine_gun,
        "The_Robot":    generate_the_robot,
        "Gaussian_Faker": generate_gaussian_faker,
        "Uniform_Jitter": generate_uniform_jitter,
        "Burst_Mode":   generate_burst_mode,
    }
    
    total_files = 0
    print("🚀 Starting dataset generation...")
    print(f"Parameters: Output Dir='{args.output_dir}', Files/Type={args.files_per_type}, Events/File={args.events_per_file}\n")
    
    for attack_name, generator_func in attack_types.items():
        print(f"Generating {attack_name}...")
        for i in range(1, args.files_per_type + 1):
            # Generate timings
            dwells, flights = generator_func(args.events_per_file)
            
            # Create filename
            filename = f"{attack_name}_{i:03d}.txt"
            
            # Write log, passing the output directory
            write_bot_log(filename, dwells, flights, args.output_dir)
            total_files += 1

    print("-" * 40)
    print(f"✅ Done! Files generated: {total_files}")
    print(f"Folder: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
