import os
import shutil
from importlib.util import find_spec

original_constants_directory = os.path.dirname(
  find_spec("craftax.craftax.constants").origin
)
TEXTURE_CACHE_FILE = os.path.join(original_constants_directory, "texture_cache.pbz2")
FULLMAP_TEXTURE_CACHE_FILE = os.path.join(
  original_constants_directory, "fullmap_texture_cache_48.pbz2"
)


def restore_cache():
  cache_dir = "craftax_cache"
  source_cache = os.path.join(cache_dir, "texture_cache.pbz2")
  source_fullmap_cache = os.path.join(cache_dir, "fullmap_texture_cache_48.pbz2")

  # Create the destination directories if they don't exist
  os.makedirs(os.path.dirname(TEXTURE_CACHE_FILE), exist_ok=True)
  os.makedirs(os.path.dirname(FULLMAP_TEXTURE_CACHE_FILE), exist_ok=True)

  # Copy the regular cache file back to the package directory
  if os.path.exists(source_cache):
    print(f"Restoring texture cache from {source_cache} to {TEXTURE_CACHE_FILE}")
    shutil.copy2(source_cache, TEXTURE_CACHE_FILE)
    print("Regular cache file restored successfully!")
  else:
    print(f"Error: Regular cache file not found in {source_cache}")
    print("Please run extract_craftax_cache.py first.")

  # Copy the fullmap cache file back to the package directory
  if os.path.exists(source_fullmap_cache):
    print(
      f"Restoring fullmap texture cache from {source_fullmap_cache} to {FULLMAP_TEXTURE_CACHE_FILE}"
    )
    shutil.copy2(source_fullmap_cache, FULLMAP_TEXTURE_CACHE_FILE)
    print("Fullmap cache file restored successfully!")
  else:
    print(f"Error: Fullmap cache file not found in {source_fullmap_cache}")
    print("Please run extract_craftax_cache.py first.")


if __name__ == "__main__":
  restore_cache()
