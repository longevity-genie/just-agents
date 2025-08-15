#!/usr/bin/env python3
"""
Simple Logging Redirection Example

This example demonstrates how to redirect agent logs from stdio to a file
using the @log_function.setter to replace the default log_print function.

Key concept: agent.log_function = custom_logger_function
"""

import datetime
import tempfile
from typing import Callable

from just_agents.base_agent import BaseAgentWithLogging
from just_agents import llm_options


def create_file_logger(log_file_path: str) -> Callable:
    """Create a custom logging function that writes to file instead of stdout."""
    def file_log_function(log_string: str, action: str, source: str, *args, **kwargs) -> None:
        shortname = kwargs.pop("agent_shortname", "")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if shortname:
            log_message = f"[{timestamp}] {action} from <{shortname}> {source}: {log_string}"
        else:
            log_message = f"[{timestamp}] {action} from {source}: {log_string}"
        
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
            f.flush()
    
    return file_log_function


def main():
    """Demonstrate logging redirection."""
    print("🔧 LOGGING REDIRECTION DEMO")
    print("=" * 50)
    
    # Create a temporary log file
    log_file_path = f"/tmp/agent_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create agent (logs will initially go to stdout)
    agent = BaseAgentWithLogging(
        llm_options=llm_options.GEMINI_2_5_FLASH,
        system_prompt="You are a helpful assistant."
    )
    
    print(f"📝 Redirecting logs to: {log_file_path}")
    
    # REDIRECT LOGS: Replace default log_print with custom file logger
    agent.log_function = create_file_logger(log_file_path)
    
    print("✅ Logs redirected! Console output is now clean.")
    print("🔍 All verbose agent messages go to the file instead.")
    
    # Test the agent (logs will go to file)
    try:
        print("\n🧪 Testing agent query...")
        response = agent.query("Say hello!")
        print(f"Agent response: {response}")
    except Exception as e:
        print(f"Query failed: {e}")
    
    # Show the log file contents
    print(f"\n📖 Contents of {log_file_path}:")
    print("-" * 40)
    try:
        with open(log_file_path, 'r') as f:
            print(f.read())
    except FileNotFoundError:
        print("(No log file found)")
    print("-" * 40)
    
    print("\n💡 Key insight: All agent debug messages are now in the file!")
    print("   This keeps your console output clean and readable.")


if __name__ == "__main__":
    main()
