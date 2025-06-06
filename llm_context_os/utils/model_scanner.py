# yawl/utils/model_scanner.py
import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import shutil # For __main__ demo cleanup

# Assuming AvailableModel is in api.schemas. If utils is a very core component,
# it might be better to define an internal structure here and convert to AvailableModel
# at a higher level (e.g., in an API handler) to decouple utils from api.schemas.
# For this task, direct import is fine as per prompt.
try:
    from yawl.api.schemas import AvailableModel
except ImportError:
    # Fallback for environments where api.schemas might not be in PYTHONPATH,
    # e.g. if running this script directly for testing without full package install.
    # This is a simplified version for the script to run.
    # In a real package, this dependency should be resolvable.
    print("Warning: Could not import AvailableModel from yawl.api.schemas. Using a local placeholder.")
    class AvailableModel: # type: ignore
        def __init__(self, model_id: str, model_type: str, path_or_identifier: str, name: Optional[str] = None,
                     description: Optional[str] = None, details: Optional[Dict[str, Any]] = None, source: str = "unknown"):
            self.model_id = model_id
            self.model_type = model_type
            self.path_or_identifier = path_or_identifier
            self.name = name or Path(path_or_identifier).name
            self.description = description
            self.details = details or {}
            self.source = source

        def model_dump_json(self, indent=None, exclude_unset=False): # Mock for Pydantic's method
            # exclude_unset is not easily implemented here without Pydantic's logic
            d = self.__dict__
            if exclude_unset: # very basic version
                d = {k:v for k,v in d.items() if v is not None}
                if not self.details: d.pop('details', None)

            return json.dumps(d, indent=indent)


def infer_model_type_from_path(path: Path) -> Optional[str]:
    """Infers the model type based on file extension or directory contents."""
    if path.is_file():
        if path.suffix == ".gguf":
            return "gguf"
        # Potentially .safetensors for single-file models, but usually these are part of a dir.
        # Add other direct file types here, e.g., .bin for older Llama.cpp models
    elif path.is_dir():
        # Check for Hugging Face style models (AWQ, EXL2, generic Transformers)
        config_json_path = path / "config.json"
        if config_json_path.exists():
            try:
                with open(config_json_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                # AWQ Check
                quant_config = config_data.get("quantization_config", {})
                if isinstance(quant_config, dict) and quant_config.get("quant_method") == "awq":
                    return "awq"

                # EXL2 Check (heuristic)
                # EXL2 often has specific files like measurement.json, out_meta.json, model.safetensors.index.json
                # and a specific "model_type" in config.json (though "model_type" can be generic like "llama")
                # A combination of these might be a stronger indicator.
                # For this example, presence of "measurement.json" is a strong hint.
                if (path / "measurement.json").exists() or (path / "out_meta.json").exists():
                     # And check for safetensors files
                    if list(path.glob("*.safetensors")):
                        return "exl2"

                # Generic Hugging Face Transformers directory
                # Check for model_type in config, e.g., "llama", "mistral", "bert"
                # Also common files like pytorch_model.bin or model.safetensors
                if config_data.get("model_type") and \
                   (list(path.glob("*.safetensors")) or list(path.glob("*.bin")) or list(path.glob("*.pt"))):
                    return "hf_transformers_dir"

            except json.JSONDecodeError:
                print(f"Warning: Could not parse config.json at {path}")
            except Exception as e:
                print(f"Warning: Error reading config.json at {path}: {e}")
            # If config.json exists but doesn't match specific types, it could still be a generic HF model.
            # Fallback to hf_transformers_dir if it has other common HF files.
            if list(path.glob("*.safetensors")) or list(path.glob("*.bin")) or list(path.glob("*.pt")):
                 return "hf_transformers_dir"


    return None

def scan_model_directories(directories_to_scan: List[str]) -> List[AvailableModel]:
    """
    Scans specified base directories for potential LLM models.
    """
    found_models: List[AvailableModel] = []
    processed_paths = set() # To avoid processing a model directory multiple times if glob hits inner files

    for base_dir_str in directories_to_scan:
        base_dir = Path(base_dir_str).resolve()
        if not base_dir.is_dir():
            print(f"Warning: Provided path {base_dir_str} is not a directory or does not exist. Skipping.")
            continue

        print(f"Scanning directory: {base_dir}")

        # Iterate over all files and directories recursively
        for item_path in base_dir.rglob('*'):
            # For directory-based models, we want to identify the main model directory, not its sub-files individually as models.
            # If item_path is a file that helps identify a directory model (like config.json),
            # we should process its parent directory.

            potential_model_path = item_path

            # If it's a common file inside a model directory, consider its parent
            if item_path.name in ["config.json", "model.safetensors.index.json", "measurement.json", "tokenizer_config.json"]:
                potential_model_path = item_path.parent

            if potential_model_path in processed_paths:
                continue

            model_type = infer_model_type_from_path(potential_model_path)

            if model_type:
                model_id = str(potential_model_path.relative_to(base_dir)) # Unique ID within this scan pass relative to base
                name = potential_model_path.name
                abs_path_str = str(potential_model_path)
                details: Dict[str, Any] = {"source_scan_directory": str(base_dir)}

                if model_type == "gguf":
                    try:
                        details["size_bytes"] = potential_model_path.stat().st_size
                    except Exception: pass
                elif model_type in ["awq", "exl2", "hf_transformers_dir"]:
                    # Try to load some basic info from config.json if present
                    config_json_path = potential_model_path / "config.json"
                    if config_json_path.exists():
                        try:
                            with open(config_json_path, 'r', encoding='utf-8') as f:
                                config_data = json.load(f)
                            if "model_type" in config_data: # HF model_type, e.g. "llama"
                                details["hf_model_type"] = config_data["model_type"]
                            if "_name_or_path" in config_data:
                                details["hf_name_or_path"] = config_data["_name_or_path"]
                                # Use HF name as description if main description is missing
                            if "architectures" in config_data:
                                details["architectures"] = config_data["architectures"]
                            if model_type == "awq": # AWQ specific details
                                details["quantization_config"] = config_data.get("quantization_config")
                        except Exception as e:
                            print(f"Could not read details from config.json for {potential_model_path}: {e}")

                # Check if this exact path_or_identifier has already been added from another base_dir scan (if multiple base_dirs overlap)
                # This check is more robust if we use a global set of added absolute paths.
                # For now, `processed_paths` handles duplicates within a single rglob pass starting from one base_dir.
                # If multiple base_dirs are given, a model could be listed multiple times if reachable from different base_dirs.
                # A final de-duplication step based on absolute path might be needed if that's an issue.

                found_models.append(AvailableModel(
                    model_id=model_id,
                    model_type=model_type,
                    path_or_identifier=abs_path_str,
                    name=name,
                    description=details.get("hf_name_or_path"), # Basic description
                    details=details,
                    source="local_scan"
                ))
                processed_paths.add(potential_model_path)
                if model_type != "gguf": # If it's a directory type, also add its contents to avoid re-processing
                    for sub_item in potential_model_path.rglob('*'):
                        processed_paths.add(sub_item)

    # De-duplicate based on absolute path_or_identifier
    final_models_dict: Dict[str, AvailableModel] = {}
    for model in found_models:
        if model.path_or_identifier not in final_models_dict:
            final_models_dict[model.path_or_identifier] = model
        else:
            # Potentially merge details if found from different relative paths, but for now, first one wins
            pass

    return list(final_models_dict.values())


if __name__ == '__main__':
    print("--- Model Scanner Demo ---")
    temp_scan_dir = Path("./temp_models_for_scan_demo")

    # Cleanup before demo if it exists
    if temp_scan_dir.exists():
        shutil.rmtree(temp_scan_dir)
    temp_scan_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy model structures
    (temp_scan_dir / "my_llama.gguf").write_text("dummy gguf content")

    awq_model_dir = temp_scan_dir / "my_awq_model_dir"
    awq_model_dir.mkdir()
    with open(awq_model_dir / "config.json", 'w') as f:
        json.dump({"quantization_config": {"quant_method": "awq"}, "model_type": "llama"}, f)
    (awq_model_dir / "pytorch_model.bin").write_text("dummy awq weights")


    exl2_model_dir = temp_scan_dir / "my_exl2_model_dir"
    exl2_model_dir.mkdir()
    with open(exl2_model_dir / "config.json", 'w') as f:
        json.dump({"model_type": "llama"}, f) # EXL2 might have generic model_type
    (exl2_model_dir / "model.safetensors.index.json").write_text("{}")
    (exl2_model_dir / "measurement.json").write_text("{}") # Key EXL2 file


    hf_model_dir = temp_scan_dir / "my_hf_model_dir"
    hf_model_dir.mkdir()
    with open(hf_model_dir / "config.json", 'w') as f:
        json.dump({"model_type": "bert", "_name_or_path": "bert-base-uncased"}, f)
    (hf_model_dir / "pytorch_model.bin").write_text("dummy hf weights")

    # Test with a nested directory
    nested_dir = temp_scan_dir / "nested_models"
    nested_dir.mkdir()
    (nested_dir / "another_model.gguf").write_text("more dummy gguf")

    print(f"\nScanning directory: {temp_scan_dir.resolve()}")
    scanned_models = scan_model_directories([str(temp_scan_dir)])

    print("\n--- Scanned Models ---")
    if scanned_models:
        for model in scanned_models:
            # Use model_dump_json if it's a real Pydantic model, else __dict__
            if hasattr(model, 'model_dump_json'):
                print(model.model_dump_json(indent=2, exclude_unset=True))
            else:
                print(json.dumps(model.__dict__, indent=2))
    else:
        print("No models found.")

    # Cleanup after demo
    if temp_scan_dir.exists():
        print(f"\nCleaning up demo directory: {temp_scan_dir}")
        shutil.rmtree(temp_scan_dir)

    print("\nModel Scanner demo complete.")
