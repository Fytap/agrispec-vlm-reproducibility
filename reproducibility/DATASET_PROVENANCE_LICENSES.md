+# Dataset provenance, licences, and redistribution boundary

This submission uses public data only. The evidence archive does not redistribute source imagery or masks.

| Dataset | Manuscript role | Version/download | Integrity evidence | Licence | Redistribution decision |
|---|---|---|---|---|---|
| SugarBeets2016 crop/weed annotations | Source foreground and crop/weed specialists; source validation diagnostics | 2016 crop/weed annotation release; downloaded 2026-08-06 | 11-part RAR, 22,950,895,611 bytes; per-part hash-table SHA-256 `424a2e9d…cb2c`; archive test exit 0 | CC BY-SA 4.0, as stated on the [official StachnissLab page](https://www.ipb.uni-bonn.de/data/sugarbeets2016/index.html); official-page snapshot SHA-256 `7bfdd146…a5e8` | Attribution/share-alike review is required; raw images and annotations are excluded from this submission archive |
| WeedsGalore | Target train/validation/official test | WACV 2025 public archive; downloaded 2026-08-05 | ZIP CRC/structure passed; 336,989,804 bytes; archive SHA-256 `5d8395c2…b68c`; no unsafe or encrypted members | Dataset CC BY 4.0; code Apache 2.0, as documented by the [official GFZ repository](https://github.com/GFZ/weedsgalore); bundled dataset-licence SHA-256 `9ba9550a…429411` | Attribution is required; raw images and masks are excluded from this submission archive |

The full 64-character hashes and machine-readable fields are in `configs/datasets.sat_revision8_public_sources_v1.yaml`. The submission includes derived non-image tables, scripts, frozen configurations, sanitized summaries, and integrity hashes. Public DINOv2 weights and all training checkpoints are acquired separately or excluded.

