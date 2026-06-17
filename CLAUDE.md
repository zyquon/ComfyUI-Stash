# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ComfyUI-Stash is a ComfyUI custom node extension that connects to [Stash](https://github.com/stashapp/stash) (a media organization server) via GraphQL API. It allows ComfyUI workflows to query and retrieve images from Stash databases using various search criteria.

## Development Commands

### Installation
```bash
# Install runtime dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

### Code Quality
```bash
# Run linter (with auto-fix)
ruff check --fix .

# Run formatter
ruff format .

# Run type checker
mypy .

# Install pre-commit hooks (runs ruff on commit)
pre-commit install
```

### Testing
```bash
# Tests and dev scripts live under tests/ (matches pyproject testpaths)
python3 tests/test_vr_face.py
```
Each file under `tests/` puts the package root on `sys.path` itself, so run it
directly from the repo root. `tests/bench_vr_remap.py` times rectify vs un-rectify
across resolutions (see its module docstring).

The VR nodes import `cv2` (opencv) and `insightface`, which are **not in `requirements.txt`** (only pillow/numpy/requests). Install them separately to run the VR face nodes.

### GraphQL Client Generation
The `stash_client/` directory contains auto-generated code from the Stash GraphQL schema using `ariadne-codegen`. To regenerate:

```bash
# Set your Stash API key
export STASH_API_KEY="your_key_here"

# Regenerate client code from queries.graphql
ariadne-codegen
```

This reads from `queries.graphql` and generates Python client code based on the remote Stash GraphQL schema.

## Architecture

### High-Level Structure

1. **Node Registration** (`__init__.py`) - Defines `NODE_CLASS_MAPPINGS` and `WEB_DIRECTORY` for ComfyUI to discover nodes
2. **Node Implementations** (`nodes.py`) - Contains two main nodes:
   - `StashNode` - Establishes connection to Stash server, returns STASH connection object
   - `StashImage` - Queries for images using ID, search string, or tags
3. **VR Face Nodes** (`vr_face.py`) - Two nodes for distortion-free face-swapping on VR/180° footage:
   - `VRFaceRectify` - Detects faces in VR/fisheye frames, selects the target face(s) (ReActor-style `input_faces_order`/`input_faces_index`/`detect_gender_input` widgets, or a wired `OPTIONS` dict which overrides them), and outputs flat face patches + a `VR_RECTIFY_MAP` (feed patches to ReActor). Selection is frame-level (one patch per selected face) because ReActor only ever sees single-face patches; default `0`/`large-small` emits just the dominant face per eye. Index `all` restores the old swap-every-face fan-out. Lazily imports `cv2`/`insightface` only when run.
   - `VRFaceUnrectify` - Projects swapped patches back into the original VR frames using the `VR_RECTIFY_MAP`
4. **Projection Geometry** (`vr_remap.py`) - Pure equirect/fisheye remap math (numpy + cv2). Has **zero ComfyUI imports on purpose** so it is unit-testable offline.
5. **Settings Management** (`settings.py`) - Reads ComfyUI user settings to get Stash API URL and key
6. **GraphQL Client** (`stash_client/`) - Auto-generated typed client for Stash API queries
7. **Frontend** (`js/`) - ComfyUI web interface extensions for settings UI
8. **Utilities** (`util.py`) - URL querystring helpers

### Data Flow

```
ComfyUI User Settings → Settings.get_settings() → StashNode creates Stash client →
StashImage uses client to query → GraphQL request to Stash →
Image retrieval via HTTP → PIL processing → Torch tensor batch → ComfyUI IMAGE
```

### Key Design Patterns

- **Settings Integration**: Uses ComfyUI's `PromptServer.instance.user_manager` to locate user settings file at `users/<user>/comfy.settings.json`. Settings are namespaced as `zyquon.ComfyUI-Stash.*`.

- **GraphQL Client**: The entire `stash_client/` directory is generated code. Never manually edit these files. Modify `queries.graphql` instead and regenerate.

- **Image Processing**: `StashImage.run()` handles multi-frame images (animated webp/GIF) by iterating with `PIL.ImageSequence.Iterator` and batching frames as torch tensors.

- **Query Combination**: Multiple search parameters (IDs, search string, tags) are OR'd together. Results are deduplicated by image ID before returning.

- **Tag Resolution**: Tag names are converted to IDs via regex query (`tags_by_regex`) to support the GraphQL `findImages` API that requires tag IDs.

### Configuration Files

- `pyproject.toml` - Project metadata, dependencies, tool configurations (mypy, ruff, pytest, ariadne-codegen)
- `.pre-commit-config.yaml` - Runs ruff linting/formatting on git commits
- `queries.graphql` - Source of truth for all GraphQL queries used by the client

### Important Implementation Details

- The Stash client requires an API key passed via `ApiKey` header
- Default Stash URL is `http://localhost:9999/graphql`
- Images are fetched via HTTP GET using the same API key header
- Empty query results return an empty torch tensor `(0, 0, 0, 3)`
- The `offset` parameter allows selecting from multiple query results (0-indexed)
- Multi-frame images must have consistent dimensions; mismatched frames are skipped with warnings

## ComfyUI Integration Notes

- Nodes are discovered via `NODE_CLASS_MAPPINGS` dictionary
- Node inputs/outputs use ComfyUI's type system: `STASH` and `VR_RECTIFY_MAP` (custom opaque types), `IMAGE`, `STRING`, `INT`
- Settings UI is defined in `js/` and registered via `WEB_DIRECTORY`
- Custom types like `STASH` are opaque Python objects passed between nodes
