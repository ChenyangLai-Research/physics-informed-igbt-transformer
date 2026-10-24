# Public-release checklist

Keep this repository private. No merge or public-release action is authorized by the current audit.

- [ ] Resolve code rights and the exact project-wide license in RELEASE_DECISIONS.md (R).
- [ ] Resolve dataset access, redistribution, schemas and identifier disclosure (D).
- [ ] Approve the limited scientific scope or supply the authoritative correction (M).
- [ ] Decide how to handle the original Git email/hostname before public release (H).
- [ ] Approve the exact archival-source disclosure scope (A).
- [ ] Confirm current PR-head CI results for Linux, macOS and Windows, including dependency advisory checks; do not rely only on a workflow definition or an older commit.
- [ ] Review all history, generated bundles, issue/PR attachments and release assets before a later visibility change.
- [ ] Obtain a separate explicit instruction to merge and/or publish after the above decisions.

The maintained entry point and archival scripts have different validation scopes. See VALIDATION.md and legacy/README.md. Input hashes detect accidental changes; they do not authenticate a checkpoint or establish data rights. No original research data or trained weights are committed.
