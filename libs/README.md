# Native Libraries

This directory contains platform-specific native libraries required for GPU acceleration.

## DirectML (Windows DirectX 12 GPU Acceleration)

### What is DirectML?
DirectML provides GPU acceleration via DirectX 12 on Windows for AMD, Intel, and NVIDIA GPUs.

### The Version Conflict Problem
- **Windows System32** includes DirectML.dll v1.8 (old, incompatible)
- **ONNX Runtime 1.22** requires DirectML.dll v1.15.4+
- Windows loads System32 DLL first → **version mismatch** → runtime errors

### Solution
Bundle DirectML v1.15.4 with the executable so it loads the correct version.

### Files
```
libs/windows/x64/DirectML.dll  (18 MB, v1.15.4)
```

### How It Works
1. **build.rs** automatically copies DirectML.dll to `target/release/` during build
2. **GitHub Actions** includes DirectML.dll in release artifacts
3. Users place DirectML.dll next to the executable

### Source
Downloaded from NuGet: [Microsoft.AI.DirectML v1.15.4](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4)

Direct download:
```bash
curl -L -o directml.nupkg "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.4"
unzip directml.nupkg -d directml_extracted
cp directml_extracted/bin/x64-win/DirectML.dll libs/windows/x64/
```

### Updating DirectML
To update to a newer version:
1. Download from: https://www.nuget.org/packages/Microsoft.AI.DirectML/
2. Extract `bin/x64-win/DirectML.dll`
3. Replace `libs/windows/x64/DirectML.dll`
4. Commit with Git LFS

### Git LFS
DirectML.dll (18 MB) is stored using Git LFS.
- Clone: `git lfs pull` to download DLL
- Without LFS: Build will fail with missing DLL warning

## Future Libraries
Additional platform-specific libraries can be added here:
- `libs/linux/x64/` - Linux CUDA/ROCm libraries
- `libs/macos/arm64/` - macOS Metal/CoreML libraries
