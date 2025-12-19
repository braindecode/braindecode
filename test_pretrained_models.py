#!/usr/bin/env python
"""
End-to-end test script for all pre-trained models on Hugging Face Hub
"""
import torch


def test_bendr():
    """Test BENDR model loading from Hub."""
    print("\n" + "=" * 70)
    print("1. Testing BENDR")
    print("=" * 70)

    from braindecode.models import BENDR

    print(
        "Loading BENDR.from_pretrained('braindecode/braindecode-bendr', n_outputs=2)..."
    )
    model = BENDR.from_pretrained("braindecode/braindecode-bendr", n_outputs=2)
    model.eval()

    x = torch.randn(2, model.n_chans, 256 * 5)  # 5 seconds at 256 Hz
    with torch.no_grad():
        out = model(x)
    print(f"  ✓ BENDR: n_chans={model.n_chans}, input={x.shape}, output={out.shape}")
    return True


def test_biot():
    """Test BIOT models loading from Hub."""
    print("\n" + "=" * 70)
    print("2. Testing BIOT models")
    print("=" * 70)

    from braindecode.models import BIOT

    biot_repos = [
        "braindecode/biot-pretrained-prest-16chs",
        "braindecode/biot-pretrained-shhs-prest-18chs",
        "braindecode/biot-pretrained-six-datasets-18chs",
    ]

    for repo in biot_repos:
        print(f"Loading BIOT.from_pretrained('{repo}')...")
        model = BIOT.from_pretrained(repo)
        model.eval()

        x = torch.randn(2, model.n_chans, model.n_times)
        with torch.no_grad():
            out = model(x)
        print(
            f"  ✓ {repo.split('/')[-1]}: n_chans={model.n_chans}, "
            f"input={x.shape}, output={out.shape}"
        )
    return True


def test_signal_jepa():
    """Test SignalJEPA models loading from Hub."""
    print("\n" + "=" * 70)
    print("3. Testing SignalJEPA models")
    print("=" * 70)

    from braindecode.models import (
        SignalJEPA,
        SignalJEPA_Contextual,
        SignalJEPA_PostLocal,
        SignalJEPA_PreLocal,
    )

    sjepa_repos = [
        ("braindecode/SignalJEPA-pretrained", SignalJEPA),
        ("braindecode/SignalJEPA-Contextual-pretrained", SignalJEPA_Contextual),
        ("braindecode/SignalJEPA-PostLocal-pretrained", SignalJEPA_PostLocal),
        ("braindecode/SignalJEPA-PreLocal-pretrained", SignalJEPA_PreLocal),
    ]

    for repo, cls in sjepa_repos:
        print(f"Loading {cls.__name__}.from_pretrained('{repo}')...")
        model = cls.from_pretrained(repo)
        model.eval()

        x = torch.randn(2, model.n_chans, model.n_times)
        with torch.no_grad():
            out = model(x)
        print(
            f"  ✓ {cls.__name__}: n_chans={model.n_chans}, n_times={model.n_times}, "
            f"input={x.shape}, output={out.shape}"
        )
    return True


def test_labram():
    """Test Labram model loading from Hub."""
    print("\n" + "=" * 70)
    print("4. Testing Labram")
    print("=" * 70)

    from braindecode.models import Labram

    print("Loading Labram.from_pretrained('braindecode/labram-pretrained')...")
    model = Labram.from_pretrained("braindecode/labram-pretrained")
    model.eval()

    x = torch.randn(2, model.n_chans, model.n_times)
    with torch.no_grad():
        out = model(x)
    print(
        f"  ✓ Labram: n_chans={model.n_chans}, n_times={model.n_times}, "
        f"input={x.shape}, output={out.shape}"
    )
    return True


def main():
    print("=" * 70)
    print("End-to-End Test: All Braindecode Pre-trained Models on Hugging Face Hub")
    print("=" * 70)

    results = {}

    # Test each model
    results["BENDR"] = test_bendr()
    results["BIOT"] = test_biot()
    results["SignalJEPA"] = test_signal_jepa()
    results["Labram"] = test_labram()

    # Summary
    print("\n" + "=" * 70)
    if all(results.values()):
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
        for name, passed in results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {name}")
    print("=" * 70)

    print(
        """
Available pre-trained models:

| Model | Repository |
|-------|------------|
| BENDR | braindecode/braindecode-bendr |
| BIOT (16ch) | braindecode/biot-pretrained-prest-16chs |
| BIOT (18ch SHHS) | braindecode/biot-pretrained-shhs-prest-18chs |
| BIOT (18ch 6-datasets) | braindecode/biot-pretrained-six-datasets-18chs |
| SignalJEPA | braindecode/SignalJEPA-pretrained |
| SignalJEPA_Contextual | braindecode/SignalJEPA-Contextual-pretrained |
| SignalJEPA_PostLocal | braindecode/SignalJEPA-PostLocal-pretrained |
| SignalJEPA_PreLocal | braindecode/SignalJEPA-PreLocal-pretrained |
| Labram | braindecode/labram-pretrained |
"""
    )


if __name__ == "__main__":
    main()
