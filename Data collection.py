from bing_image_downloader import downloader
import os
import shutil

def download_images(queries, category, output_base, limit):
    for query in queries:
        print(f"[INFO] Downloading: '{query}' → category: {category}")
        downloader.download(
            query,
            limit=limit,
            output_dir=output_base,
            adult_filter_off=True,
            force_replace=False,
            timeout=60,
            verbose=True
        )

        # Move images into category folder
        query_folder = os.path.join(output_base, query)
        target_folder = os.path.join(output_base, category)

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        if os.path.exists(query_folder):
            for file in os.listdir(query_folder):
                src = os.path.join(query_folder, file)
                dst = os.path.join(target_folder, file)
                if os.path.isfile(src):
                    shutil.move(src, dst)
            shutil.rmtree(query_folder)  # Remove empty query folder


def collect_image_dataset():
    output_dir = 'D:/vislona/dataset'
    limit_per_query = 105

    # Queries must be lists, not strings
    good_queries = [
        "good quality images",
        "clear sharp images",
        "high resolution photo"
    ]

    bad_queries = [
        "bad quality images",
        "blurry photo",
        "low resolution image",
        "dark photo",
        "overexposed picture"
    ]

    print("\n--- Collecting GOOD images ---")
    download_images(good_queries, "good", output_dir, limit_per_query)

    print("\n--- Collecting BAD images ---")
    download_images(bad_queries, "bad", output_dir, limit_per_query)

    print("\n✅ Dataset download complete.")
    print(f"Check folders: '{output_dir}/good' and '{output_dir}/bad'")
    print("🧹 Tip: Still review manually to clean logos, watermarks, irrelevant images.")


if _name_ == '_main_':
    collect_image_dataset()
