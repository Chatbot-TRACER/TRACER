"""Utilities for saving user profiles."""

from pathlib import Path

import yaml

from tracer.utils.logging_utils import get_logger

logger = get_logger()


def save_profiles(built_profiles: list[dict], output_dir: str) -> None:
    """Save user profiles as YAML files in the specified directory.

    Args:
        built_profiles: List of dictionaries representing user profiles
        output_dir: Directory to write the profile files to
    """
    if not built_profiles:
        logger.info("No user profiles to save")
        return

    # Create profiles subdirectory
    profiles_dir = Path(output_dir) / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    error_count = 0
    for profile in built_profiles:
        test_name = profile.get("test_name", f"profile_{hash(str(profile))}")

        if isinstance(test_name, dict):
            if test_name.get("function") == "random()" and "data" in test_name and test_name["data"]:
                base_name = str(test_name["data"][0])
                filename_base = f"random_profile_{base_name.lower().replace(' ', '_')}"
            else:
                filename_base = f"profile_{hash(str(test_name))}"
        else:
            filename_base = str(test_name).lower().replace(" ", "_").replace(",", "").replace("&", "and")

        safe_filename_base = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in filename_base)
        filename = f"{safe_filename_base}.yaml"
        filepath = profiles_dir / filename

        try:
            with filepath.open("w", encoding="utf-8") as yf:
                yaml.dump(
                    profile,
                    yf,
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                    width=1000,
                )
            saved_count += 1
            logger.debug("Saved profile: %s", filename)
        except yaml.YAMLError:
            logger.exception("Failed to create YAML for profile '%s'.", test_name)
            error_count += 1
        except OSError:
            logger.exception("Failed to write file '%s'.", filename)
            error_count += 1

    if error_count:
        logger.warning("Saved %d profiles with %d errors", saved_count, error_count)
    else:
        logger.info("Successfully saved %d profiles to: %s/", saved_count, profiles_dir)
