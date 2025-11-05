#!/usr/bin/env python
"""
Interactive Hugging Face login and test script.
"""

import sys

def login_to_hf():
    """Login to Hugging Face Hub."""
    print("\n" + "=" * 70)
    print("LOGGING IN TO HUGGING FACE HUB")
    print("=" * 70)

    try:
        from huggingface_hub import login, whoami

        print("\nOption 1: Login with token")
        print("  Get your token from: https://huggingface.co/settings/tokens")
        print("\nOption 2: Login interactively (will open browser)")
        print("\nPress Enter to continue with interactive login...")
        input()

        # Try interactive login
        print("\n✓ Opening browser for authentication...")
        login()

        # Verify login
        print("\n✓ Verifying login...")
        user_info = whoami()
        username = user_info['name']
        print(f"✅ Successfully logged in as: {username}")

        return username

    except Exception as e:
        print(f"\n❌ Login failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have a Hugging Face account")
        print("2. Visit: https://huggingface.co/join")
        print("3. Get a token from: https://huggingface.co/settings/tokens")
        print("4. Run: from huggingface_hub import login; login(token='your-token')")
        return None


def main():
    """Main function."""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 12 + "HUGGING FACE HUB - LOGIN & TEST" + " " * 24 + "║")
    print("╚" + "=" * 68 + "╝")

    # Step 1: Login
    username = login_to_hf()
    if not username:
        return 1

    # Step 2: Suggest test repository name
    print("\n" + "=" * 70)
    print("READY TO TEST")
    print("=" * 70)
    print(f"\nSuggested repository name: {username}/test-eegnet-braindecode")
    print("\nTo run the full test, execute:")
    print(f"  python test_hf_hub_integration.py {username}/test-eegnet-braindecode")
    print("\nOr press Enter to run it now...")

    response = input().strip()
    if response.lower() not in ['n', 'no', 'skip']:
        import subprocess
        repo_name = f"{username}/test-eegnet-braindecode"
        print(f"\n✓ Running test with repository: {repo_name}")
        result = subprocess.run(
            [sys.executable, "test_hf_hub_integration.py", repo_name],
            capture_output=False
        )
        return result.returncode

    return 0


if __name__ == "__main__":
    exit(main())
