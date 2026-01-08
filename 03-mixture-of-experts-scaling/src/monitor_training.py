import subprocess
import time
import datetime

def monitor_gpu():
    """Monitor GPU usage during training"""
    print("GPU Monitoring (Press Ctrl+C to stop)")
    print("=" * 80)
    
    try:
        while True:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                data = result.stdout.strip()
                print(f"[{timestamp}] {data}")
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == '__main__':
    monitor_gpu()