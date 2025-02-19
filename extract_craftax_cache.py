import os
import shutil
from craftax.craftax.constants import TEXTURE_CACHE_FILE
from craftax_fullmap_constants import TEXTURE_CACHE_FILE as FULLMAP_TEXTURE_CACHE_FILE


def extract_cache():
  # Create cache directory if it doesn't exist
  cache_dir = "craftax_cache"
  os.makedirs(cache_dir, exist_ok=True)

  # Copy the regular texture cache file if it exists
  if os.path.exists(TEXTURE_CACHE_FILE):
    print(f"Copying texture cache from: {TEXTURE_CACHE_FILE}")
    shutil.copy2(TEXTURE_CACHE_FILE, os.path.join(cache_dir, "texture_cache.pbz2"))
    print("Regular cache file extracted successfully!")
  else:
    print(
      "Warning: Regular texture cache file not found. Run craftax once to generate it first."
    )

  # Copy the fullmap texture cache file if it exists
  if os.path.exists(FULLMAP_TEXTURE_CACHE_FILE):
    print(f"Copying fullmap texture cache from: {FULLMAP_TEXTURE_CACHE_FILE}")
    shutil.copy2(
      FULLMAP_TEXTURE_CACHE_FILE,
      os.path.join(cache_dir, "fullmap_texture_cache_48.pbz2"),
    )
    print("Fullmap cache file extracted successfully!")
  else:
    print(
      "Warning: Fullmap texture cache file not found. Run craftax with fullmap once to generate it first."
    )


if __name__ == "__main__":
  extract_cache()
