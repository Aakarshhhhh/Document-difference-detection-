#!/usr/bin/env python3
"""
Startup script for the Document Difference Detection app.
"""
import subprocess
import sys
import os

def run_app():
    """Run the Streamlit app with proper configuration"""
    print("🚀 Starting Document Difference Detection App...")
    print("📁 Current directory:", os.getcwd())
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found in current directory!")
        return False
    
    # Configure Streamlit settings
    streamlit_config = [
        "--server.port", "8501",
        "--server.address", "localhost",
        "--server.headless", "false",
        "--browser.serverAddress", "localhost",
        "--browser.gatherUsageStats", "false",
        "--server.enableCORS", "false"
    ]
    
    try:
        print("\n🌐 Starting Streamlit server...")
        print("   Local URL: http://localhost:8501")
        print("   Press Ctrl+C to stop the server")
        print("\n" + "="*50)
        
        # Run streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py"] + streamlit_config
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Server stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running Streamlit: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = run_app()
    if success:
        print("👋 Thanks for using Document Difference Detection!")
    else:
        print("💥 App encountered an error. Please check the logs above.")
        sys.exit(1)