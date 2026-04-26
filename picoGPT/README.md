# picoGPT (Reference Baseline)

This directory contains the original, minimalist GPT-2 implementation by [Jay Mody](https://github.com/jaymody/picoGPT). It serves as the **Reference Baseline** for our EML-native port.

## Why this baseline?
`picoGPT` is an unnecessarily tiny and minimal implementation of GPT-2 in plain NumPy. Because it uses standard operators (division, sqrt, softmax), it exhibits "multiplicative fragility" (NaN spikes) that we solve in our EML-native version.

## Usage
Run the standard baseline inference:
```bash
python gpt2.py "Alan Turing theorized that computers would one day become"
```

For the EML-native version with formal verification support, see the [**eml-picogpt/**](../eml-picogpt/) directory.
