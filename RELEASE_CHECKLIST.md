# Public-release checklist

The curated entry point and the archival scripts have different validation scopes.
Keep this repository private until the unresolved items below have been reviewed.

- [ ] Confirm permission from the relevant authors, institutions and partners for the selected code and embedded scientific constants.
- [ ] Select and add a code license only after rights are confirmed. No license is granted by the current repository.
- [ ] Confirm access and redistribution terms for each original dataset. No research dataset is included; synthetic examples do not establish data rights.
- [ ] Review archival filenames, authorship headers and experimental settings for confidential information. Preserve required attribution.
- [ ] Check all Git history before changing visibility. Removing a file from the latest revision does not remove earlier copies.
- [ ] Confirm the current publication status and the match between this implementation and the cited study.
- [ ] Review the independent Linux/Windows CI results before claiming compatibility. Local validation covers macOS arm64 CPU only.
- [ ] Resolve scientific limitations in README.md before claiming paper reproduction or deploying the models in practice.

Use only trusted checkpoints. Never commit local output bundles or original data without a separate disclosure review.

The original commit metadata contains a local machine email/hostname. Audit commits use the GitHub noreply address; original history still needs a privacy decision before public release.
